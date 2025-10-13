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
from typing import Any, Dict, List, Optional, Tuple

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import wandb
import jaxpruner
from jaxpruner import api
from purejaxql.sparse_training_api import create_sparse_updater
from jaxpruner import utils
import envpool
import ml_collections
from functools import partial
from purejaxql.utils.atari_wrapper import (
    JaxLogEnvPoolWrapper, 
    Transition, 
    CustomTrainState,
    eps_greedy_exploration,
)
from purejaxql.networks import create_network
import purejaxql.logging_utils as analysis

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
        network = create_network(config, env.single_action_space.n)

        def create_agent(rng):
            rng, rng_sparse = jax.random.split(rng, 2)
            init_x = jnp.zeros((1, *env.single_observation_space.shape))
            network_variables = network.init(rng, init_x, train=False)
            sparse_config = create_jaxpruner_config(config)
            sparse_config.rng_seed = rng_sparse
            pruner = create_sparse_updater(sparse_config)
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
                perturbations=network_variables["perturbations"],
            )
            return train_state, post_op

        rng, _rng = jax.random.split(rng)
        train_state, post_op = create_agent(rng)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, expl_state, test_metrics, cumulative_mask, rng = runner_state

            # SAMPLE PHASE: This runs one env step for all parallel envs and returns the transition at that step.
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                q_vals, _ = network.apply(
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

                # TODO: Transition like object could be used to story states for policy churn.
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

            # TODO: what is this used for?
            last_q, _ = network.apply(
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

            rng, sample_rng = jax.random.split(rng)
            full_batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
            flat_transitions = jax.tree_util.tree_map(
                lambda x: x.reshape(full_batch_size, *x.shape[2:]),
                transitions
            )
            flat_targets = lambda_targets.reshape(full_batch_size)
            sample_size = config["ANALYSIS_KWARGS"]["ANALYSIS_SAMPLE_SIZE"]
            indices = jax.random.choice(sample_rng, full_batch_size, shape=(sample_size,), replace=False)

            analysis_transitions_sampled = jax.tree_util.tree_map(lambda x: x[indices], flat_transitions)
            analysis_targets_sampled = flat_targets[indices]
            analysis_minibatch = Transition(
                obs=analysis_transitions_sampled.obs,
                action=analysis_transitions_sampled.action,
                reward=analysis_transitions_sampled.reward,
                done=analysis_transitions_sampled.done,
                next_obs=analysis_transitions_sampled.next_obs,
                q_val=analysis_transitions_sampled.q_val,
            )
            func_to_shape = partial(
                analysis.compute_combined_dormancy_metrics,
                network=network,
                config=config,
                action_dim=env.single_action_space.n
            )
            
            dummy_old_params = jax.tree_util.tree_map(lambda x: x, train_state.params)
            dummy_old_batch_stats = jax.tree_util.tree_map(lambda x: x, train_state.batch_stats) 
            dummy_metrics_tree = jax.eval_shape(
                func_to_shape,
                params=train_state.params,
                old_params=dummy_old_params,
                batch_stats=train_state.batch_stats,
                old_batch_stats=dummy_old_batch_stats,
                perturbations=train_state.perturbations,
                minibatch=analysis_minibatch, # Use the sampled minibatch for shape inference
                target=analysis_targets_sampled,
            )

            dummy_metrics_tree = jax.tree_util.tree_map(
                lambda x: jnp.zeros(x.shape, x.dtype),
                dummy_metrics_tree
            )
            # dummy_metrics_tree = analysis.create_dummy_analysis_metrics(config)

            def _compute_analysis_on_batch(state_data_and_old_state):
                ts, mb, t, old_p, old_bs = state_data_and_old_state
                return analysis.compute_combined_dormancy_metrics(
                    network,
                    ts.params,
                    old_p,
                    ts.batch_stats,
                    old_bs,
                    ts.perturbations,
                    mb,
                    t,
                    config,
                    env.single_action_space.n
                )

            def _no_op_analysis(_):
                return dummy_metrics_tree
            
            
            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, cumulative_mask, rng = carry

                def _learn_phase(carry, minibatch_and_target):

                    train_state, cumulative_mask, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):
                        (q_vals, _), updates = network.apply(
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

                        return loss, (updates, chosen_action_qvals, q_vals)

                    (loss, (updates, qvals, full_q)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    post_params = post_op(train_state.params, train_state.opt_state)
                    train_state = train_state.replace(
                        params=post_params,
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    gn = optax.global_norm(grads)
                    cumulative_mask = jax.tree_util.tree_map(
                        lambda c, m: jnp.logical_or(c, m).astype(jnp.uint8), # <-- Cast the result back
                        cumulative_mask,
                        train_state.opt_state.masks,
                    )
                    return (train_state, cumulative_mask, rng), (loss, qvals, full_q, gn) #, analysis_metrics)

                # This is just to shuffle the minibatches
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
                (train_state, cumulative_mask, rng), (loss, qvals, full_q, gn) = jax.lax.scan(
                    _learn_phase, (train_state, cumulative_mask, rng), (minibatches, targets)
                )

                return (train_state, cumulative_mask, rng), (loss, qvals, full_q, gn) #, analysis_metrics)

            old_params = jax.tree_util.tree_map(lambda x: x, train_state.params)
            old_batch_stats = jax.tree_util.tree_map(lambda x: x, train_state.batch_stats)
            
            rng, _rng = jax.random.split(rng)
            (train_state, cumulative_mask, rng), (loss, qvals, full_q, gn) = jax.lax.scan(
                _learn_epoch, (train_state, cumulative_mask, rng), None, config["NUM_EPOCHS"] # <- Increasing Num expochs increases num gradient steps... this is where I could leverage extended training?
            )
            
            should_log_analysis_batch = (train_state.n_updates % config["ANALYSIS_KWARGS"]["ANALYSIS_BATCH_PERIOD"]) == 0
            analysis_metrics = jax.lax.cond(
                should_log_analysis_batch,
                _compute_analysis_on_batch,
                _no_op_analysis,
                operand=(train_state, analysis_minibatch, analysis_targets_sampled, old_params, old_batch_stats) # Pass the sampled data
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
            explored_params = jax.tree_util.tree_reduce(
                jnp.add, jax.tree_util.tree_map(jnp.sum, cumulative_mask)
            )
            total_params = jax.tree_util.tree_reduce(jnp.add, jax.tree_util.tree_map(lambda x: x.size, train_state.opt_state.masks))
            exploration_coefficient = explored_params / total_params
            metrics = {
                "training/env_step": train_state.timesteps,
                "training/update_steps": train_state.n_updates,
                "env_frame": train_state.timesteps
                * env.observation_space.shape[
                    0
                ],  # first dimension of the observation space is number of stacked frames
                "training/grad_steps": train_state.grad_steps,
                "training/td_loss": loss.mean(),
                "q_values/qvals": qvals.mean(),
                # "q_values/QVarience": jnp.var(jnp.max(full_q, axis=-1)),
                "q_values/QNorm(l2)": jnp.linalg.norm(qvals, axis=-1).mean(),
                "target/QVarience": jnp.var(lambda_targets),
                "sparsity/param_sparsity": param_sparsity['_total_sparsity'],
                "sparsity/mask_sparsity": mask_sparsity['_total_sparsity'],
                "sparsity/exploration_coefficient": exploration_coefficient,
                "params/param_norm(l2)": optax.global_norm(train_state.params),
                "grads/grad_norm(l2)": gn.mean(),
            }

            metrics.update({k: v.mean() for k, v in infos.items()})
            if config.get("TEST_DURING_TRAINING", False):
                metrics.update({f"test/{k}": v.mean() for k, v in test_infos.items()})

            # report on wandb if required
            # if config["WANDB_MODE"] != "disabled":

                # The new, simplified callback
            def callback(metrics, analysis_metrics, original_seed):
                # This part is unchanged
                metrics = jax.tree_util.tree_map(np.asarray, metrics)

                if config.get("WANDB_LOG_ALL_SEEDS", False):
                    metrics.update({f"rng{int(original_seed)}/{k}": v for k, v in metrics.items()})
                wandb.log(metrics)

                # --- START OF CHANGES ---

                # 1. No more complex reshaping or looping.
                # 2. A simple check to see if the analysis was run in this update.
                #    The dummy metric tree we created has srank=0, so this works perfectly.
                # print(analysis_metrics['ranks/srank_kumar'])
                if analysis_metrics['ranks/srank_kumar'] > 0:

                    # 3. Restructure and log the single metric dictionary.
                    restructured_log = analysis.restructure_single_step_metrics(analysis_metrics)

                    final_log = {}
                    for key, val in restructured_log.items():
                        final_log[key] = val

                    # 4. Log against the current gradient step for consistent plotting.
                    # final_log['training/grad_steps'] = metrics["training/grad_steps"]
                    final_log = jax.tree_util.tree_map(np.asarray, final_log)
                    wandb.log(final_log)

            # The call to the callback is also simpler, as analysis_metrics is now a single, top-level variable.
            jax.debug.callback(callback, metrics, analysis_metrics, original_seed)

            runner_state = (train_state, tuple(expl_state), test_metrics, cumulative_mask, rng)

            return runner_state, metrics

        # test metrics not supported yet
        test_metrics = None

        # train
        rng, _rng = jax.random.split(rng)
        expl_state = (init_obs, env_state)
        
        cumulative_mask_init = jax.tree_util.tree_map(
            jnp.zeros_like, train_state.opt_state.masks
        )

        runner_state = (train_state, expl_state, test_metrics, cumulative_mask_init, _rng)

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
        name=f'{config["ALG_NAME"]}_{config["ENV_NAME"]}_{config["SEED"]}_{config["PRUNER_KWARGS"]["pruner"]}_{config["PRUNER_KWARGS"]["sparsity"]}', # TODO: add jaxpruner info to run name as well
        config=config,
        mode=config["WANDB_MODE"],
    )
    wandb.define_metric("training/update_steps")
    wandb.define_metric("*", step_metric="training/update_steps")
    # wandb.define_metric("training/grad_steps")
    # wandb.define_metric("analysis/*", step_metric="training/grad_steps")
    
    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    if config["NUM_SEEDS"] > 1:
        raise NotImplementedError("Vmapped seeds not supported yet.") # can't do parallel seeds yet
    else:
        outs = jax.jit(make_train(config))(rng)
        # outs = make_train(config)(rng) # don't jit for now, as it takes too long to compile
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
