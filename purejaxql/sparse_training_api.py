import os
import argparse
import dataclasses
import functools
import math
from functools import partial
from typing import Any, Sequence, Tuple, Optional, Dict, Callable, Type

# Configure JAX for minimal memory usage - allocate only what's needed (optional)
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# --- Third-party imports ---
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
# import tensorflow as tf
# import tensorflow_datasets as tfds
import chex
import copy
import logging

from jaxpruner import algorithms
from jaxpruner import base_updater
from jaxpruner import sparsity_distributions
from jaxpruner import sparsity_schedules
from jaxpruner import sparsity_types
import ml_collections

import wandb  # type: ignore
# jaxpruner imports
import jaxpruner
from jaxpruner.algorithms.sparse_trainers import * 
from jaxpruner.algorithms.sparse_trainers import restart_inner_state
from jaxpruner.algorithms import pruners
from jaxpruner.algorithms.sparse_trainers import SET as BaseSET
from jaxpruner.algorithms.sparse_trainers import RigL as BaseRigL  
from jaxpruner import mask_calculator
from jaxpruner import base_updater
from jaxpruner import sparsity_schedules 

# Hide the GPU from tensorflow so it doesn't pre-allocate memory
# (JAX will use the accelerator instead.)
# tf.config.experimental.set_visible_devices([], "GPU")



###################### Helper functions ################################

import dataclasses
import logging

import chex
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class EarlyTrainingPeriodicSchedule(sparsity_schedules.PeriodicSchedule):
  """Implements periodic update schedule for early training."""
  def start_sparse_training_iter(self, step):
    
    diff = step - self.update_start_step # NOTE: this is because the update state function get's called when step%update_freq == 0 and > start step
    return diff < self.update_freq 
  
  def is_mask_update_iter(self, step):
    if step is None:
      return False
    is_update_step = jnp.logical_and(
        step % self.update_freq == 0, step >= self.update_start_step
    )
    if self.update_end_step is not None:
      is_update_step = jnp.logical_and(
          is_update_step, step <= self.update_end_step
      )
    return is_update_step

@dataclasses.dataclass(frozen=True)
class PolynomialSchedule_GMP_DST(sparsity_schedules.PolynomialSchedule):
  def is_mask_update_iter(self, step):
    if step is None:
      return False
    is_update_step = jnp.logical_and(
        step % self.update_freq == 0, step >= self.update_start_step
    )
    return is_update_step
  
  def is_dst_iter(self, step):
    if step is None:
      return False
    is_update_step = jnp.logical_and(
        step % self.update_freq == 0, step >= self.update_end_step
    )
    return is_update_step

  
@dataclasses.dataclass(frozen=True)
class PolynomialScheduleDST(sparsity_schedules.PolynomialSchedule):
  
  update_freq_dst: int = 1000
  
  def is_mask_update_iter(self, step):
    if step is None:
      return False
    is_st_update_step = jnp.logical_and(
        step % self.update_freq_dst == 0, step >= self.update_start_step + self.update_freq
    )
    # if self.update_end_step is not None:
    #   is_st_update_step = jnp.logical_and(
    #       is_st_update_step, step <= self.update_end_step
    #   )
    return is_st_update_step

  def is_mask_update_iter_gmp(self, step):
    if step is None:
      return False
    is_update_step = jnp.logical_and(
        step % self.update_freq == 0, step >= self.update_start_step
    )
    if self.update_end_step is not None:
      is_update_step = jnp.logical_and(
          is_update_step, step <= self.update_end_step
      )
    return is_update_step

######################### Custom Algorithms Registry ##########################


################# Custom Sparse Trainers ###################################################
@dataclasses.dataclass
class RandomSET(BaseSET):
    def _get_drop_scores(self, sparse_state, params, grads):
        new_rng = jax.random.fold_in(self.rng_seed, sparse_state.count)
        random_scores = pruners.generate_random_scores(params, new_rng)
        random_scores = self.apply_masks(random_scores, sparse_state.masks)
        return jax.tree.map(jnp.abs, random_scores)  # TODO: See if needed


@dataclasses.dataclass
class GradSET(BaseSET):
    def _get_drop_scores(self, sparse_state, params, grads):
        return jax.tree.map(jnp.abs, grads)

@dataclasses.dataclass
class GMP_DST(BaseSET):
  
  def get_initial_masks(
      self, params, target_sparsities
      ):
        """Generate initial mask. This is only used when .wrap_optax() is called."""

        def mask_fn(p, target):
          if target is None:
            return None
          return jnp.ones(p.shape, dtype=mask_calculator.MASK_DTYPE)

        masks = jax.tree_map(mask_fn, params, target_sparsities)
        return masks
  
  def calculate_scores(self, params, sparse_state=None, grads=None):
    del sparse_state, grads
    param_magnitudes = jax.tree_map(jnp.abs, params)
    return param_magnitudes
  
  def update_state_dst(self, sparse_state, params, grads):
    # jax.debug.print("[DEBUG]: Executing DST update at step {count}", count=sparse_state.count)

    drop_scores = self._get_drop_scores(sparse_state, params, grads)
    grow_scores = self._get_grow_scores(sparse_state, params, grads)
    current_drop_fraction = self.drop_fraction_fn(
        sparse_state.count - self.scheduler.update_end_step + 1
    )
    # jax.debug.print("[DEBUG]: Drop fraction at step {drop_fr}", drop_fr=current_drop_fraction)

    update_masks_fn = functools.partial(
        self._update_masks, drop_fraction=current_drop_fraction
    )
    new_masks = jax.tree_map(
        update_masks_fn, sparse_state.masks, drop_scores, grow_scores
    )

    masks_activated = jax.tree_map(
        lambda old_mask, new_mask: (old_mask == 0) & (new_mask == 1),
        sparse_state.masks,
        new_masks,
    )
    new_inner_state = restart_inner_state(
        sparse_state.inner_state, masks_activated
    )
    return sparse_state._replace(masks=new_masks, inner_state=new_inner_state)

  
  def update_state_gmp(self, sparse_state, params, grads):
    # jax.debug.print("[DEBUG]: Executing GMP update at step {count}", count=sparse_state.count)
    
    
    sparsities = self.scheduler.get_sparsity_at_step(
        sparse_state.target_sparsities, sparse_state.count
    )
    scores = self.calculate_scores(
        params, sparse_state=sparse_state, grads=grads)
    new_masks = self.create_masks(scores, sparsities)
    if self.use_packed_masks:
      new_masks = jax.tree_map(jnp.packbits, new_masks)
    return sparse_state._replace(masks=new_masks)
  
  def update_state(self, sparse_state, params, grads):
    """Updates the mask tree and returns a dict of metrics."""
    return jax.lax.cond(
        self.scheduler.is_dst_iter(sparse_state.count),
        lambda _: self.update_state_dst(sparse_state, params, grads),
        lambda _: self.update_state_gmp(sparse_state, params, grads),
        operand=None
    )

@dataclasses.dataclass
class SET_rl(BaseSET):
  
  def get_initial_masks(
      self, params, target_sparsities
      ):
        """Generate initial mask. This is only used when .wrap_optax() is called."""

        def mask_fn(p, target):
          if target is None:
            return None
          return jnp.ones(p.shape, dtype=mask_calculator.MASK_DTYPE)

        masks = jax.tree_map(mask_fn, params, target_sparsities)
        return masks
  
  def update_initial_state(self, sparse_state, params, grads):
    del grads
    jax.debug.print(">>> Initial mask created <<< {}", sparse_state.count)
    scores = jax.tree_map(jnp.abs, params)
    new_masks = self.create_masks(scores, sparse_state.target_sparsities)
    return sparse_state._replace(masks=new_masks)
  
  
  def _update_state(self, sparse_state, params, grads):
    """Updates the mask tree and returns a dict of metrics."""
    # jax.debug.print(">>> Entered DST <<< {}", sparse_state.count)

    drop_scores = self._get_drop_scores(sparse_state, params, grads)
    grow_scores = self._get_grow_scores(sparse_state, params, grads)
    current_drop_fraction = self.drop_fraction_fn(sparse_state.count)
    update_masks_fn = functools.partial(
        self._update_masks, drop_fraction=current_drop_fraction
    )
    new_masks = jax.tree_map(
        update_masks_fn, sparse_state.masks, drop_scores, grow_scores
    )
    masks_activated = jax.tree_map(
        lambda old_mask, new_mask: (old_mask == 0) & (new_mask == 1),
        sparse_state.masks,
        new_masks,
    )
    new_inner_state = restart_inner_state(
        sparse_state.inner_state, masks_activated
    )
    return sparse_state._replace(masks=new_masks, inner_state=new_inner_state)
    
  def update_state(self, sparse_state, params, grads):
    """Updates the mask tree and returns a dict of metrics."""
    # jax.debug.print("Start sparse training: {}", self.scheduler.start_sparse_training_iter(sparse_state.count))
    return jax.lax.cond(
        self.scheduler.start_sparse_training_iter(sparse_state.count),
        lambda _: self.update_initial_state(sparse_state, params, grads),
        lambda _: self._update_state(sparse_state, params, grads),
        operand=None
    )


@dataclasses.dataclass
class RigL_rl(SET_rl):
  
  def _get_grow_scores(self, sparse_state, params, grads):
    del sparse_state, params
    return jax.tree_map(jnp.abs, grads)


@dataclasses.dataclass
class Gradual_SET(BaseSET):
    scheduler: Any = PolynomialScheduleDST
    """
    Implements a gradual RigL approach, switching between RigL and GMP updates.
    """
    def get_initial_masks(
      self, params, target_sparsities
      ):
        """Generate initial mask. This is only used when .wrap_optax() is called."""

        def mask_fn(p, target):
          if target is None:
            return None
          return jnp.ones(p.shape, dtype=mask_calculator.MASK_DTYPE)

        masks = jax.tree_map(mask_fn, params, target_sparsities)
        return masks

    def calculate_scores(self, params, sparse_state=None, grads=None):
        del sparse_state, grads
        param_magnitudes = jax.tree_map(jnp.abs, params)
        return param_magnitudes

    def update_SET_state(self, sparse_state, params, grads, **kwargs):
        """Updates the state using RigL logic."""
        drop_scores = self._get_drop_scores(sparse_state, params, grads)
        grow_scores = self._get_grow_scores(sparse_state, params, grads)
        current_drop_fraction = self.drop_fraction_fn(sparse_state.count)
        # jax.debug.print("checking dropfraction at step {df}", df=current_drop_fraction)

        update_masks_fn = functools.partial(
            self._update_masks, drop_fraction=current_drop_fraction
        )
        new_masks = jax.tree_map(
            update_masks_fn, sparse_state.masks, drop_scores, grow_scores
        )
        masks_activated = jax.tree_map(
            lambda old_mask, new_mask: (old_mask == 0) & (new_mask == 1),
            sparse_state.masks,
            new_masks,
        )
        new_inner_state = restart_inner_state(
            sparse_state.inner_state, masks_activated
        )
        return sparse_state._replace(masks=new_masks, inner_state=new_inner_state)

    def update_GMP_state(self, sparse_state, params, grads, **kwargs):
        """Updates the state using GMP logic."""
        sparsities = self.scheduler.get_sparsity_at_step(
            sparse_state.target_sparsities, sparse_state.count
        )
        scores = self.calculate_scores(
            params, sparse_state=sparse_state, grads=grads, **kwargs
        )
        new_masks = self.create_masks(scores, sparsities)
        if self.use_packed_masks:
            new_masks = jax.tree_map(jnp.packbits, new_masks)
        return sparse_state._replace(masks=new_masks)

    def update_state(self, sparse_state, params, grads, **kwargs):
        """
        Determines whether to use RigL or GMP update logic, and updates accordingly.
        """

        def set_update(sparse_state, params, grads, **kwargs):
            # jax.debug.print("Executing RigL update at step {count}", count=sparse_state.count)
            return self.update_SET_state(sparse_state, params, grads)

        def gmp_update(sparse_state, params, grads, **kwargs):
            # jax.debug.print("Executing GMP update at step {count}", count=sparse_state.count)
            return self.update_GMP_state(sparse_state, params, grads, **kwargs)

        # Use jax.lax.cond for control flow with a traced boolean.
        return jax.lax.cond(self.scheduler.is_mask_update_iter_gmp(sparse_state.count), gmp_update, set_update, sparse_state, params, grads, **kwargs)


@dataclasses.dataclass
class Gradual_RigL(BaseRigL):
    scheduler: Any = PolynomialScheduleDST
    """
    Implements a gradual RigL approach, switching between RigL and GMP updates.
    """
    def get_initial_masks(
      self, params, target_sparsities
      ):
        """Generate initial mask. This is only used when .wrap_optax() is called."""

        def mask_fn(p, target):
          if target is None:
            return None
          return jnp.ones(p.shape, dtype=mask_calculator.MASK_DTYPE)

        masks = jax.tree_map(mask_fn, params, target_sparsities)
        return masks

    def calculate_scores(self, params, sparse_state=None, grads=None):
        del sparse_state, grads
        param_magnitudes = jax.tree_map(jnp.abs, params)
        return param_magnitudes

    def update_RigL_state(self, sparse_state, params, grads, **kwargs):
        """Updates the state using RigL logic."""
        drop_scores = self._get_drop_scores(sparse_state, params, grads)
        grow_scores = self._get_grow_scores(sparse_state, params, grads)
        current_drop_fraction = self.drop_fraction_fn(sparse_state.count)

        update_masks_fn = functools.partial(
            self._update_masks, drop_fraction=current_drop_fraction
        )
        new_masks = jax.tree_map(
            update_masks_fn, sparse_state.masks, drop_scores, grow_scores
        )
        masks_activated = jax.tree_map(
            lambda old_mask, new_mask: (old_mask == 0) & (new_mask == 1),
            sparse_state.masks,
            new_masks,
        )
        new_inner_state = restart_inner_state(
            sparse_state.inner_state, masks_activated
        )
        return sparse_state._replace(masks=new_masks, inner_state=new_inner_state)

    def update_GMP_state(self, sparse_state, params, grads, **kwargs):
        """Updates the state using GMP logic."""
        sparsities = self.scheduler.get_sparsity_at_step(
            sparse_state.target_sparsities, sparse_state.count
        )
        scores = self.calculate_scores(
            params, sparse_state=sparse_state, grads=grads, **kwargs
        )
        new_masks = self.create_masks(scores, sparsities)
        if self.use_packed_masks:
            new_masks = jax.tree_map(jnp.packbits, new_masks)
        return sparse_state._replace(masks=new_masks)

    def update_state(self, sparse_state, params, grads, **kwargs):
        """
        Determines whether to use RigL or GMP update logic, and updates accordingly.
        """

        def rigl_update(sparse_state, params, grads, **kwargs):
            # jax.debug.print("Executing RigL update at step {count}", count=sparse_state.count)
            return self.update_RigL_state(sparse_state, params, grads)

        def gmp_update(sparse_state, params, grads, **kwargs):
            # jax.debug.print("Executing GMP update at step {count}", count=sparse_state.count)
            return self.update_GMP_state(sparse_state, params, grads, **kwargs)

        # Use jax.lax.cond for control flow with a traced boolean.
        return jax.lax.cond(self.scheduler.is_mask_update_iter_gmp(sparse_state.count), gmp_update, rigl_update, sparse_state, params, grads, **kwargs)


@dataclasses.dataclass
class GraNet(BaseRigL):
    def get_initial_masks(
        self, params, target_sparsities
        ):
          """Generate initial mask. This is only used when .wrap_optax() is called."""

          def mask_fn(p, target):
            if target is None:
              return None
            return jnp.ones(p.shape, dtype=mask_calculator.MASK_DTYPE)

          masks = jax.tree_map(mask_fn, params, target_sparsities)
          return masks
    
    def calculate_scores(self, params, sparse_state=None, grads=None):
        del sparse_state, grads
        param_magnitudes = jax.tree_map(jnp.abs, params)
        return param_magnitudes
      
    def update_state(self, sparse_state, params, grads, **kwargs):
        # Magnitude Pruning
        sparsities = self.scheduler.get_sparsity_at_step(
            sparse_state.target_sparsities, sparse_state.count
        )
        scores = self.calculate_scores(
            params, sparse_state=sparse_state, grads=grads, **kwargs
        )
        old_mask = self.create_masks(scores, sparsities)
        if self.use_packed_masks:
            old_mask = jax.tree_map(jnp.packbits, old_mask)

        # Prune more and Regenerate
        drop_scores = self._get_drop_scores(sparse_state, params, grads)
        grow_scores = self._get_grow_scores(sparse_state, params, grads)
        current_drop_fraction = self.drop_fraction_fn(sparse_state.count)
        update_masks_fn = functools.partial(
            self._update_masks, drop_fraction=current_drop_fraction
        )
        new_masks = jax.tree_map(
            update_masks_fn, old_mask, drop_scores, grow_scores
        )
        masks_activated = jax.tree_map(
            lambda old_mask, new_mask: (old_mask == 0) & (new_mask == 1),
            sparse_state.masks,
            new_masks,
        )
        new_inner_state = restart_inner_state(
            sparse_state.inner_state, masks_activated
        )
        return sparse_state._replace(masks=new_masks, inner_state=new_inner_state)
      
@dataclasses.dataclass
class RandomGraNet(GraNet):

    def _get_grow_scores(self, sparse_state, params, grads):
        new_rng = jax.random.fold_in(self.rng_seed, sparse_state.count)
        random_scores = pruners.generate_random_scores(params, new_rng)
        return random_scores

#################################################################################

CUSTOM_ALGORITHM_REGISTRY = {
    "rset": RandomSET,
    "rrigl": None,
    "gradset": GradSET,
    'gradual_rigl': Gradual_RigL,
    'gradual_set': Gradual_SET,
    'granet': GraNet,
    'random_granet': RandomGraNet,
    'gmp_dst': GMP_DST,
    'set_rl': SET_rl,
    'rigl_rl': RigL_rl,
}

CUSTOM_ALGORITHMS = tuple(CUSTOM_ALGORITHM_REGISTRY.keys())

# Copied from jaxpruner api.py
def create_updater_from_custom_config(
    sparsity_config,
):
    """Gets a sparsity updater based on the given sparsity config.

    A sample usage of this api is in below.
    ```
      sparsity_config = ml_collections.ConfigDict()
      # Required
      sparsity_config.algorithm = 'magnitude'
      sparsity_config.dist_type = 'erk'
      sparsity_config.sparsity = 0.8

      # Optional
      sparsity_config.update_freq = 10
      sparsity_config.update_end_step = 1000
      sparsity_config.update_start_step = 200
      sparsity_config.sparsity_type = 'nm_2,4'

      updater = create_updater_from_config(sparsity_config)
    ```

    - `algorithm`: str, one of jaxpruner.all_algorithm_names()
    - `dist_type`: str, 'erk' or 'uniform'.
    - `update_freq`: int, passed to PeriodicSchedule.
    - `update_end_step`: int, passed to PeriodicSchedule.
    - `update_start_step`: int, if None or doesn't exist NoUpdateSchedule is
      used. If equal to `update_end_step`, OneShotSchedule is used. Otherwise
      PolynomialSchedule is used.
    - `sparsity`: str, float or jaxpruner.SparsityType, if float, then
      SparsityType.Unstructured is used. If str in '{N}:{M}' format,
      then SparsityType.NbyM is used. If str in '{N}x{M}' format,
      then SparsityType.Block is used. User can also pass the desired
      SparsityType directly. Everything else under `sparsity_config`
      is passed to the algorithm directly.

    Args:
      sparsity_config: configuration for the algorithm. See options above.

    Returns:
      an algorithm with the given configuration.
    """
    logging.info("Creating  updater for %s", sparsity_config.algorithm)

    config = copy.deepcopy(sparsity_config).unlock()

    if config.dist_type == "uniform":
        config.sparsity_distribution_fn = sparsity_distributions.uniform
    elif config.dist_type == "erk":
        config.sparsity_distribution_fn = sparsity_distributions.erk
    else:
        raise ValueError(
            f"dist_type: {config.dist_type} is not supported. " "Use `erk` or `uniform`"
        )
    del config.dist_type

    if config.get("filter_fn", None):
        if not config.algorithm.startswith("global_"):
            new_fn = functools.partial(
                config.sparsity_distribution_fn, filter_fn=config.filter_fn
            )
            config.sparsity_distribution_fn = new_fn
            del config.filter_fn

    if config.get("custom_sparsity_map", None):
        if not config.algorithm.startswith("global_"):
            new_fn = functools.partial(
                config.sparsity_distribution_fn,
                custom_sparsity_map=config.custom_sparsity_map,
            )
            config.sparsity_distribution_fn = new_fn
            del config.custom_sparsity_map

    if config.algorithm.startswith("global_"):
        # Distribution function is not used.
        del config.sparsity_distribution_fn
    else:
        kwargs = {"sparsity": config.sparsity}
        del config.sparsity
        if config.get("filter_fn", None):
            kwargs["filter_fn"] = config.filter_fn
            del config.filter_fn
        config.sparsity_distribution_fn = functools.partial(
            config.sparsity_distribution_fn, **kwargs
        )
    if config.get("sparsity_type", None):
        s_type = config.sparsity_type
        if isinstance(s_type, str) and s_type.startswith("nm"):
            # example: nm_2,4
            n, m = s_type.split("_")[1].strip().split(",")
            del config.sparsity_type
            config.sparsity_type = sparsity_types.NByM(int(n), int(m))
        elif isinstance(s_type, str) and (s_type.startswith("block")):
            # example: block_4,4
            n, m = s_type.split("_")[1].strip().split(",")
            del config.sparsity_type
            config.sparsity_type = sparsity_types.Block(block_shape=(int(n), int(m)))
        elif isinstance(s_type, str) and (s_type.startswith("channel")):
            axis = int(s_type.split("_")[1])
            del config.sparsity_type
            config.sparsity_type = sparsity_types.Channel(axis=axis)
        else:
            raise ValueError(f"Sparsity type {s_type} is not supported.")

    if config.algorithm in CUSTOM_ALGORITHM_REGISTRY:
        updater_type = CUSTOM_ALGORITHM_REGISTRY[config.algorithm]
        if config.algorithm in ("rrigl", "rset", "gradset"):
            config.drop_fraction_fn = optax.cosine_decay_schedule(
                config.get("drop_fraction", 0.1), config.update_end_step
            )
            config.scheduler = sparsity_schedules.PolynomialSchedule(
                update_freq=config.update_freq,
                update_start_step=config.update_start_step,
                update_end_step=config.update_end_step,
            )
            
        # del config.algorithm
        elif config.algorithm in ("rigl_rl", "set_rl"):
            config.drop_fraction_fn = optax.cosine_decay_schedule(
                config.get("drop_fraction", 0.1), config.update_end_step
            )
            config.scheduler = EarlyTrainingPeriodicSchedule(
                update_freq=config.update_freq,
                update_start_step=config.update_start_step,
                update_end_step=config.update_end_step,
            )
        
        elif config.algorithm in ("gradual_rigl", "gradual_set"):
            config.drop_fraction_fn = optax.cosine_decay_schedule(
                config.get("drop_fraction", 0.1), config.update_end_step
            )
            config.scheduler = PolynomialScheduleDST(
                update_freq=config.update_freq,
                update_freq_dst=config.update_freq // 3,
                update_start_step=config.update_start_step,
                update_end_step=config.update_end_step,
            )
        elif config.algorithm in ("gmp_dst",):
            config.drop_fraction_fn = optax.cosine_decay_schedule(
                config.get("drop_fraction", 0.1), config.update_end_step
            )
            config.scheduler = PolynomialSchedule_GMP_DST(
                update_freq=config.update_freq,
                update_start_step=config.update_start_step,
                update_end_step=config.update_end_step,
            )
        
        elif config.algorithm in ("granet", "random_granet"):
            config.drop_fraction_fn = optax.cosine_decay_schedule(
                config.get("drop_fraction", 0.1), config.update_end_step
            )
            config.scheduler = sparsity_schedules.PolynomialSchedule(
                update_freq=config.update_freq,
                update_start_step=config.update_start_step,
                update_end_step=config.update_end_step,
            )
        
    else:
        raise ValueError(
            f"Sparsity algorithm {config.algorithm} is not supported."
            " Please use a key from jaxpruner.all_algorithm_names() or register"
            " the new algorithm using jaxpruner.register_algorithm()."
        )

    # if config.get("update_start_step", None) is None:
    #     config.scheduler = sparsity_schedules.NoUpdateSchedule()
    # elif config.update_end_step == config.update_start_step:
    #     config.scheduler = sparsity_schedules.OneShotSchedule(
    #         target_step=config.update_end_step
    #     )
    # else:
    #     config.scheduler = sparsity_schedules.PolynomialSchedule(
    #         update_freq=config.update_freq,
    #         update_start_step=config.update_start_step,
    #         update_end_step=config.update_end_step,
    #     )
    del config.algorithm
    for field_name in (
        "update_freq",
        "update_start_step",
        "update_end_step",
        "drop_fraction",
    ):
        if hasattr(config, field_name):
            delattr(config, field_name)

    updater = updater_type(**config)

    return updater


def create_sparse_updater(sparsity_config):

    if sparsity_config.algorithm in CUSTOM_ALGORITHM_REGISTRY:
        return create_updater_from_custom_config(sparsity_config)
    else:
        return jaxpruner.api.create_updater_from_config(sparsity_config)
