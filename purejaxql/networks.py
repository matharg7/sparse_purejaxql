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


class CNNQNetwork(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        x = jnp.transpose(x, (0, 2, 3, 1))
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)
            x = x / 255.0
        x = CNN(norm_type=self.norm_type)(x, train)
        x = nn.Dense(self.action_dim)(x)
        return x

import jax.numpy as jnp
from flax import linen as nn

class Stack(nn.Module):
    """max-pool + two residual conv blocks, like Dopamine's Stack, WITH norm and perturbations."""
    num_ch: int
    num_blocks: int = 2
    use_max_pooling: bool = True
    use_layer_norm: bool = True  # <-- CHANGED from norm_type
    stack_idx: int = 0

    @nn.compact
    def __call__(self, x, *, train: bool):
        init = nn.initializers.xavier_uniform()

        def normalize(y):
            # Conditionally apply LayerNorm based on the flag
            if self.use_layer_norm: # <-- CHANGED
                return nn.LayerNorm()(y)
            # If the flag is false, the input is returned unchanged
            return y

        x = nn.Conv(self.num_ch, kernel_size=(3,3), strides=(1,1), padding="SAME", kernel_init=init)(x)
        # Note: I'm leaving your perturbation calls as they are.
        x = self.perturb(f'stack_{self.stack_idx}_conv_pre_norm', x)

        if self.use_max_pooling:
            x = nn.max_pool(x, window_shape=(3,3), strides=(2,2), padding="SAME")

        for block_idx in range(self.num_blocks):
            skip = x
            h = nn.relu(x)
            h = self.perturb(f'stack_{self.stack_idx}_block_{block_idx}_post_relu1', h)

            h = nn.Conv(self.num_ch, kernel_size=(3,3), strides=(1,1), padding="SAME", kernel_init=init)(h)
            h = self.perturb(f'stack_{self.stack_idx}_block_{block_idx}_conv1', h)
            h = normalize(h) # Apply conditional normalization

            h = nn.relu(h)
            h = self.perturb(f'stack_{self.stack_idx}_block_{block_idx}_post_relu2', h)

            h = nn.Conv(self.num_ch, kernel_size=(3,3), strides=(1,1), padding="SAME", kernel_init=init)(h)
            h = self.perturb(f'stack_{self.stack_idx}_block_{block_idx}_conv2', h)
            h = normalize(h) # Apply conditional normalization

            x = skip + h
        return x


class ImpalaEncoder(nn.Module):
    nn_scale: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    use_layer_norm: bool = True  # <-- CHANGED

    def setup(self):
        self.stacks = [
            Stack(
                num_ch=s * self.nn_scale, 
                num_blocks=self.num_blocks, 
                use_layer_norm=self.use_layer_norm, # <-- CHANGED
                stack_idx=i
            )
            for i, s in enumerate(self.stack_sizes)
        ]

    @nn.compact
    def __call__(self, x, *, train: bool):
        for stack_idx, s in enumerate(self.stacks):
            x = s(x, train=train)
            x = self.perturb(f'encoder_stack_{stack_idx}_output', x)
        
        x = nn.relu(x)
        x = self.perturb('encoder_final_post_relu', x)
        return x


class ImpalaBackbone(nn.Module):
    nn_scale: int = 1
    final_dim: int = 512
    inputs_preprocessed: bool = False
    use_layer_norm: bool = True  # <-- CHANGED

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        init = nn.initializers.xavier_uniform()

        if not self.inputs_preprocessed:
            _ = nn.BatchNorm(use_running_average=not train)(x)
            x = x.astype(jnp.float32) / 255.0
        else:
            x = nn.BatchNorm(use_running_average=not train)(x)

        # Pass the flag down to the encoder
        x = ImpalaEncoder(nn_scale=self.nn_scale, use_layer_norm=self.use_layer_norm)(x, train=train) # <-- CHANGED
        x = self.perturb('backbone_encoder_output', x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(self.final_dim * self.nn_scale, kernel_init=init)(x)
        x = self.perturb('backbone_dense_pre_norm', x)

        # Normalization based on the flag
        if self.use_layer_norm: # <-- CHANGED
            x = nn.LayerNorm()(x)

        x = nn.relu(x)
        x = self.perturb('backbone_final_features_post_relu', x)
        return x


class ImpalaQNetwork(nn.Module):
    action_dim: int
    use_layer_norm: bool = True  # <-- CHANGED
    impala_scale: int = 1
    impala_final_dim: int = 512
    impala_inputs_preprocessed: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Pass the flag down to the backbone
        feats = ImpalaBackbone(
            nn_scale=self.impala_scale,
            final_dim=self.impala_final_dim,
            inputs_preprocessed=self.impala_inputs_preprocessed,
            use_layer_norm=self.use_layer_norm, # <-- CHANGED
        )(x, train=train)
        
        q = nn.Dense(self.action_dim)(feats)
        return q, feats


def create_network(config, action_dim):
    # This function now expects a boolean in the config
    if config.get("NETWORK_NAME") == "impala":
        return ImpalaQNetwork(
            action_dim=action_dim,
            use_layer_norm=config.get("USE_LAYER_NORM", True), # <-- CHANGED
            impala_scale=config.get("NETWORK_WIDTH", 1),                    
            impala_final_dim=512,           
            impala_inputs_preprocessed=False,  
        )
    # The 'cnn' part of your factory function would need a similar change
    elif config.get("NETWORK_NAME") == "cnn":
        return CNNQNetwork(
            action_dim=action_dim,
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
        )
    else:
        raise ValueError(f"Unknown network name: {config.get('NETWORK_NAME')}")




