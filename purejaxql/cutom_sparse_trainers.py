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
from purejaxql.sparse_training_api import EarlyTrainingPeriodicSchedule, PolynomialScheduleDST, PolynomialSchedule_GMP_DST
# Hide the GPU from tensorflow so it doesn't pre-allocate memory
# (JAX will use the accelerator instead.)
# tf.config.experimental.set_visible_devices([], "GPU")

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