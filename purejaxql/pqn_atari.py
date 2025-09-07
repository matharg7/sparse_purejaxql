"""
When test_during_training is set to True, an additional number of parallel test environments are used to evaluate the agent during training using greedy actions, 
but not for training purposes. Stopping training for evaluation can be very expensive, as an episode in Atari can last for hundreds of thousands of steps.
"""

import copy
import time
import os
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import wandb
import jaxpruner
from jaxpruner import api
from jaxpruner import utils
import envpool
import ml_collections
from functools import partial
from purejaxql.utils.atari_wrapper import JaxLogEnvPoolWrapper


class CNN(nn.Module):

    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=nn.initializers.he_normal())(x)
        x = normalize(x)
        x = nn.relu(x)
        return x


# class QNetwork(nn.Module):
#     action_dim: int
#     norm_type: str = "layer_norm"
#     norm_input: bool = False

#     @nn.compact
#     def __call__(self, x: jnp.ndarray, train: bool):
#         x = jnp.transpose(x, (0, 2, 3, 1))
#         if self.norm_input:
#             x = nn.BatchNorm(use_running_average=not train)(x)
#         else:
#             # dummy normalize input for global compatibility
#             x_dummy = nn.BatchNorm(use_running_average=not train)(x)
#             x = x / 255.0
#         x = CNN(norm_type=self.norm_type)(x, train)
#         x = nn.Dense(self.action_dim)(x)
#         return x

# --- Dopamine-style IMPALA blocks (Flax) ---

class Stack(nn.Module):
    """max-pool + two residual conv blocks, like Dopamine's Stack, WITH norm."""
    num_ch: int
    num_blocks: int = 2
    use_max_pooling: bool = True
    norm_type: str = "layer_norm"  # <--- add

    @nn.compact
    def __call__(self, x, *, train: bool):
        init = nn.initializers.xavier_uniform()

        def normalize(y):
            if self.norm_type == "layer_norm":
                return nn.LayerNorm()(y)
            elif self.norm_type == "batch_norm":
                return nn.BatchNorm(use_running_average=not train)(y)
            else:
                return y

        # 3x3 conv then optional 3x3 max-pool stride 2
        x = nn.Conv(self.num_ch, kernel_size=(3,3), strides=(1,1),
                    padding="SAME", kernel_init=init)(x)
        x = normalize(x)                # <--- norm just like your CNN
        x = nn.relu(x)
        if self.use_max_pooling:
            x = nn.max_pool(x, window_shape=(3,3), strides=(2,2), padding="SAME")

        # Two residual blocks: (ReLU -> Conv -> Norm) x2 + skip
        for _ in range(self.num_blocks):
            skip = x
            h = nn.relu(x)
            h = nn.Conv(self.num_ch, kernel_size=(3,3), strides=(1,1),
                        padding="SAME", kernel_init=init)(h)
            h = normalize(h)            # <--- norm
            h = nn.relu(h)
            h = nn.Conv(self.num_ch, kernel_size=(3,3), strides=(1,1),
                        padding="SAME", kernel_init=init)(h)
            h = normalize(h)            # <--- norm
            x = skip + h
        return x



class ImpalaEncoder(nn.Module):
    nn_scale: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    norm_type: str = "layer_norm"   # <--- add

    def setup(self):
        self.stacks = [
            Stack(num_ch=s * self.nn_scale, num_blocks=self.num_blocks, norm_type=self.norm_type)
            for s in self.stack_sizes
        ]

    @nn.compact
    def __call__(self, x, *, train: bool):
        for s in self.stacks:
            x = s(x, train=train)
        return nn.relu(x)



class ImpalaBackbone(nn.Module):
    """
    Full Dopamine-style IMPALA body + dense(512).
    Matches: xavier_uniform in convs and dense, ReLU, /255.0 preprocessing.
    """
    nn_scale: int = 1
    final_dim: int = 512
    inputs_preprocessed: bool = False  # False => divide by 255.0 like Dopamine

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        init = nn.initializers.xavier_uniform()

        # Create a no-op BN to keep a batch_stats collection (for your TrainState)
        if not self.inputs_preprocessed:
            _ = nn.BatchNorm(use_running_average=not train)(x)  # ignored output
            x = x.astype(jnp.float32) / 255.0
        else:
            x = nn.BatchNorm(use_running_average=not train)(x)

        x = ImpalaEncoder(nn_scale=self.nn_scale)(x, train=train)
        x = x.reshape((x.shape[0], -1))              # flatten per-batch
        x = nn.Dense(self.final_dim, kernel_init=init)(x)
        x = nn.relu(x)
        return x


class ImpalaBackbone(nn.Module):
    nn_scale: int = 1
    final_dim: int = 512
    inputs_preprocessed: bool = False
    norm_type: str = "layer_norm"   # <--- add

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        init = nn.initializers.xavier_uniform()

        # Keep a BN to ensure batch_stats collection exists regardless of norm_type
        if not self.inputs_preprocessed:
            _ = nn.BatchNorm(use_running_average=not train)(x)
            x = x.astype(jnp.float32) / 255.0
        else:
            x = nn.BatchNorm(use_running_average=not train)(x)

        x = ImpalaEncoder(nn_scale=self.nn_scale, norm_type=self.norm_type)(x, train=train)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.final_dim, kernel_init=init)(x)

        # same normalization pattern as your CNN after the dense
        if self.norm_type == "layer_norm":
            x = nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            x = nn.BatchNorm(use_running_average=not train)(x)

        x = nn.relu(x)
        return x


class QNetwork(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False

    backbone: str = "impala_dopa_exact"
    impala_scale: int = 1
    impala_final_dim: int = 512
    impala_inputs_preprocessed: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        x = jnp.transpose(x, (0, 2, 3, 1))

        if self.backbone.lower() == "impala_dopa_exact":
            feats = ImpalaBackbone(
                nn_scale=self.impala_scale,
                final_dim=self.impala_final_dim,
                inputs_preprocessed=self.impala_inputs_preprocessed,
                norm_type=self.norm_type,                        # <--- pass through
            )(x, train=train)
            init = nn.initializers.xavier_uniform()
            q = nn.Dense(self.action_dim, kernel_init=init)(feats)
            return q

        # fallback Nature-CNN (unchanged)
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            _ = nn.BatchNorm(use_running_average=not train)(x)
            x = x / 255.0
        feats = CNN(norm_type=self.norm_type)(x, train)
        q = nn.Dense(self.action_dim)(feats)
        return q




@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    # How many full outer iterations can I run if I want exactly 
    # B = NUM_ENVS * NUM_STEPS transitions per update?” This is similar to constructing the entire Dataset
    
    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    # Same idea but for schedules (ε / LR). 
    # You often want the decay horizon to be decoupled from the total training budget:
    
    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    def make_env(num_envs):
        env = envpool.make(
            config["ENV_NAME"],
            env_type="gym",
            num_envs=num_envs,
            seed=config["SEED"],
            **config["ENV_KWARGS"],
        )
        env.num_envs = num_envs
        env.single_action_space = env.action_space
        env.single_observation_space = env.observation_space
        env.name = config["ENV_NAME"]
        env = JaxLogEnvPoolWrapper(env)
        return env

    total_envs = (
        (config["NUM_ENVS"] + config["TEST_ENVS"])
        if config.get("TEST_DURING_TRAINING", False)
        else config["NUM_ENVS"]
    )
    env = make_env(total_envs) # create all envs at once (test + train)

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),  # sample random actions,
            greedy_actions,
        )
        return chosed_actions

    # here reset must be out of vmap and jit
    init_obs, env_state = env.reset()

    def train(rng):

        original_seed = rng[0]

        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        # INIT NETWORK AND OPTIMIZER
        # network = QNetwork(
        #     action_dim=env.single_action_space.n,
        #     norm_type=config["NORM_TYPE"],
        #     norm_input=config.get("NORM_INPUT", False),
        # )
        network = QNetwork(
            action_dim=env.single_action_space.n,
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            backbone="impala_dopa_exact",          # <-- add
            impala_scale=1,                     # <-- add
            impala_final_dim=512,           # <-- add
            impala_inputs_preprocessed=False,  # <-- add
        )

        def create_agent(rng):
            rng, rng_sparse = jax.random.split(rng, 2)
            init_x = jnp.zeros((1, *env.single_observation_space.shape))
            network_variables = network.init(rng, init_x, train=False)
            # TODO: Add updater from jaxpruner
            sparse_config = create_jaxpruner_config(config)
            sparse_config.rng_seed = rng_sparse
            pruner = api.create_updater_from_config(
                sparse_config
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )
            post_op = jax.jit(pruner.post_gradient_update)
            tx = pruner.wrap_optax(tx)

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )
            return train_state, post_op

        rng, _rng = jax.random.split(rng)
        train_state, post_op = create_agent(rng)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, expl_state, test_metrics, rng = runner_state

            # SAMPLE PHASE: This runs one env step for all parallel envs and returns the transition at that step.
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                # TODO: Add post gradient update on params 
                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )

                # different eps for each env
                _rngs = jax.random.split(rng_a, total_envs)
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                if config.get("TEST_DURING_TRAINING", False):
                    eps = jnp.concatenate((eps, jnp.zeros(config["TEST_ENVS"])))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = env.step(
                    env_state, new_action
                )

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
                )
                return (new_obs, new_env_state, rng), (transition, info)

            # step the env
            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            if config.get("TEST_DURING_TRAINING", False):
                # remove testing envs
                transitions = jax.tree_util.tree_map(
                    lambda x: x[:, : -config["TEST_ENVS"]], transitions
                )

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            last_q = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                transitions.next_obs[-1],
                train=False,
            )
            last_q = jnp.max(last_q, axis=-1)

            def _compute_targets(last_q, q_vals, reward, done):
                def _get_target(lambda_returns_and_next_q, rew_q_done):
                    reward, q, done = rew_q_done
                    lambda_returns, next_q = lambda_returns_and_next_q
                    target_bootstrap = reward + config["GAMMA"] * (1 - done) * next_q
                    delta = lambda_returns - next_q
                    lambda_returns = (
                        target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                    )
                    lambda_returns = (1 - done) * lambda_returns + done * reward
                    next_q = jnp.max(q, axis=-1)
                    return (lambda_returns, next_q), lambda_returns

                lambda_returns = reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
                last_q = jnp.max(q_vals[-1], axis=-1)
                _, targets = jax.lax.scan(
                    _get_target,
                    (lambda_returns, last_q),
                    jax.tree_util.tree_map(lambda x: x[:-1], (reward, q_vals, done)),
                    reverse=True,
                )
                targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
                return targets

            lambda_targets = _compute_targets(
                last_q, transitions.q_val, transitions.reward, transitions.done
            )

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):

                    train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):
                        q_vals, updates = network.apply(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            minibatch.obs,
                            train=True,
                            mutable=["batch_stats"],
                        )  # (batch_size*2, num_actions)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        return loss, (updates, chosen_action_qvals)

                    (loss, (updates, qvals)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    post_params = post_op(train_state.params, train_state.opt_state)
                    train_state = train_state.replace(
                        params=post_params,
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    
                    return (train_state, rng), (loss, qvals)

                def preprocess_transition(x, rng):
                    x = x.reshape(
                        -1, *x.shape[2:]
                    )  # num_steps*num_envs (batch_size), ...
                    x = jax.random.permutation(rng, x)  # shuffle the transitions
                    x = x.reshape(
                        config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                    )  # num_mini_updates, batch_size/num_mini_updates, ...
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), transitions
                )  # num_actors*num_envs (batch_size), ...
                targets = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), lambda_targets
                )

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
                )
                

                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )
            post_params = post_op(train_state.params, train_state.opt_state)
            train_state = train_state.replace(params=post_params)
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            if config.get("TEST_DURING_TRAINING", False):
                test_infos = jax.tree_util.tree_map(lambda x: x[:, -config["TEST_ENVS"] :], infos)
                infos = jax.tree_util.tree_map(lambda x: x[:, : -config["TEST_ENVS"]], infos)
                infos.update({"test/" + k: v for k, v in test_infos.items()})

            # Added logging for differnt analysis metrics.
            param_sparsity = utils.summarize_sparsity(train_state.params, only_total_sparsity=True)
            mask_sparsity = utils.summarize_sparsity(train_state.opt_state.masks, only_total_sparsity=True)
            
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "env_frame": train_state.timesteps
                * env.observation_space.shape[
                    0
                ],  # first dimension of the observation space is number of stacked frames
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
                "param_sparsity": param_sparsity['_total_sparsity'],
                "mask_sparsity": mask_sparsity['_total_sparsity'],
            }
            

            metrics.update({k: v.mean() for k, v in infos.items()})
            if config.get("TEST_DURING_TRAINING", False):
                metrics.update({f"test/{k}": v.mean() for k, v in test_infos.items()})

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_seed)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics, step=metrics["update_steps"])

                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (train_state, tuple(expl_state), test_metrics, rng)

            return runner_state, metrics

        # test metrics not supported yet
        test_metrics = None

        # train
        rng, _rng = jax.random.split(rng)
        expl_state = (init_obs, env_state)
        runner_state = (train_state, expl_state, test_metrics, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def single_run(config):

    config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "pqn")
    env_name = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f'{config["ALG_NAME"]}_{config["ENV_NAME"]}', # TODO: add jaxpruner info to run name as well
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    if config["NUM_SEEDS"] > 1:
        raise NotImplementedError("Vmapped seeds not supported yet.") # can't do parallel seeds yet
    else:
        outs = jax.jit(make_train(config))(rng)
    print(f"Took {time.time()-t0} seconds to complete.")

    # save params < -----------------------------
    if config.get("SAVE_PATH", None) is not None:

        from purejaxql.utils.save_load import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        # assumes not vmpapped seeds
        params = model_state.params
        save_path = os.path.join(
            save_dir,
            f'{alg_name}_{env_name}_seed{config["SEED"]}.safetensors',
        )
        save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {
        **default_config,
        **default_config["alg"],
    }  # merge the alg config with the main config

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        # update the default params
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config["alg"][k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])

        if config["NUM_SEEDS"] > 1:
            raise NotImplementedError("Vmapped seeds not supported yet.")
        else:
            outs = jax.jit(make_train(config))(rng)

    sweep_config = {
        "name": f"pqn_atari_{default_config['ENV_NAME']}",
        "method": "bayes",
        "metric": {
            "name": "test_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {"values": [0.0005, 0.0001, 0.00005]},
            "LAMBDA": {"values": [0.3, 0.6, 0.9]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]: # could be used for differnt seeds too
        tune(config)
    else:
        single_run(config)

def create_jaxpruner_config(config):
    jaxpruner_config = ml_collections.ConfigDict()
    jaxpruner_config.algorithm = config["PRUNER_KWARGS"]["pruner"]
    jaxpruner_config.sparsity = config["PRUNER_KWARGS"]["sparsity"]
    jaxpruner_config.update_freq = config["PRUNER_KWARGS"]["update_frequency"]
    jaxpruner_config.update_start_step = config["PRUNER_KWARGS"]["start_step"]
    jaxpruner_config.update_end_step = config["PRUNER_KWARGS"]["end_step"]
    jaxpruner_config.dist_type = config["PRUNER_KWARGS"]["sparsity_distribution"]
    jaxpruner_config.drop_fraction = config["PRUNER_KWARGS"]["drop_fraction"]
    return jaxpruner_config

if __name__ == "__main__":
    main() # Args are loaded with the config file in side main