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




# --- Dopamine-style IMPALA blocks (Flax) ---

class Stack(nn.Module):
    """max-pool + two residual conv blocks, like Dopamine's Stack, WITH norm."""
    num_ch: int
    num_blocks: int = 2
    use_max_pooling: bool = True
    norm_type: str = "layer_norm"  # <--- add

    @nn.compact
    def __call__(self, x, *, train: bool):
        init = nn.initializers.he_normal()

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
        x = normalize(x)                
        x = nn.relu(x)
        if self.use_max_pooling:
            x = nn.max_pool(x, window_shape=(3,3), strides=(2,2), padding="SAME")

        for _ in range(self.num_blocks):
            skip = x
            h = nn.relu(x)
            h = nn.Conv(self.num_ch, kernel_size=(3,3), strides=(1,1),
                        padding="SAME", kernel_init=init)(h)
            h = normalize(h)            
            h = nn.relu(h)
            h = nn.Conv(self.num_ch, kernel_size=(3,3), strides=(1,1),
                        padding="SAME", kernel_init=init)(h)
            h = normalize(h)           
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
    nn_scale: int = 1
    final_dim: int = 512
    inputs_preprocessed: bool = False
    norm_type: str = "layer_norm"   # <--- add

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        init = nn.initializers.he_normal()

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


class ImpalaQNetwork(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False
    impala_scale: int = 1
    impala_final_dim: int = 512
    impala_inputs_preprocessed: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        x = jnp.transpose(x, (0, 2, 3, 1))
        feats = ImpalaBackbone(
            nn_scale=self.impala_scale,
            final_dim=self.impala_final_dim,
            inputs_preprocessed=self.impala_inputs_preprocessed,
            norm_type=self.norm_type,                        # <--- pass through
        )(x, train=train)
        # init = nn.initializers.xavier_uniform()
        # q = nn.Dense(self.action_dim, kernel_init=init)(feats)
        q = nn.Dense(self.action_dim)(feats)   # <--- simpler init
        return q
    
    
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
    
    
def create_network(config, action_dim):
    if config.get("NETWORK_NAME") == "cnn":
        return CNNQNetwork(
            action_dim=action_dim,
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
        )
            
    elif config.get("NETWORK_NAME") == "impala":
        return ImpalaQNetwork(
            action_dim=action_dim,
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),     
            impala_scale=config.get("NETWORK_WIDTH", 1),                    
            impala_final_dim=512,           
            impala_inputs_preprocessed=False,  
        )
    else:
        raise ValueError(f"Unknown network name: {config.get('NETWORK_NAME')}")