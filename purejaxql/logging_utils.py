"""
All implementations exactly follow paper definitions.
"""

from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jaxpruner import utils
from purejaxql.utils.atari_wrapper import Transition, CustomTrainState
import optax
import jax
import jax.numpy as jnp
from functools import partial
from jax.tree_util import tree_flatten_with_path, DictKey, SequenceKey
from typing import Any, Dict, List, Tuple, Optional

def compute_q_estimation_norm(qvals: jnp.ndarray) -> float:
    """Average L2 norm of Q(s,·) vectors (QNorm)"""
    # q = qvals.reshape(-1, qvals.shape[-1])
    norms = jnp.linalg.norm(qvals, axis=-1)
    return jnp.mean(norms)
            
def compute_parameter_l2(params) -> jnp.ndarray:
    return optax.global_norm(params)

# @partial(jax.jit, static_argnames=('tau',))
# def _compute_srank(feats: jnp.ndarray, tau: float) -> jnp.ndarray:
#     """JIT-compiled function to compute the Stable Rank (S-Rank)."""
#     feats = feats.reshape(feats.shape[0], -1)
#     s = jnp.linalg.svd(feats, full_matrices=False, compute_uv=False)
#     s0 = jnp.where(s.size > 0, s[0], jnp.array(0.0, s.dtype))
#     cutoff = jnp.asarray(tau, s.dtype) * jnp.maximum(s0, jnp.asarray(1e-12, s.dtype))
#     return jnp.sum(s >= cutoff).astype(jnp.int32)

@partial(jax.jit, static_argnames=('tau',))
def _compute_srank(feats: jnp.ndarray, tau: float) -> jnp.ndarray:
    """
    Computes the Effective Rank (srank) based on the paper definition:
    https://openreview.net/forum?id=O9bnihsFfXU
    
    srank_δ(Φ) = min{k : (Σ_{i=1 to k} σ_i) / (Σ_{i=1 to d} σ_i) ≥ 1 - δ}
    
    Here, 'tau' corresponds to the paper's 'δ'.
    """
    # Ensure feats is a 2D matrix for SVD
    feats = feats.reshape(feats.shape[0], -1)

    # 1. Get the singular values
    s = jnp.linalg.svd(feats, full_matrices=False, compute_uv=False)

    def calculate_rank():
        # 2. Calculate the total sum of singular values
        total_sum = jnp.sum(s)
        
        # 3. Calculate the cumulative sum and normalize it
        cumulative_sum = jnp.cumsum(s)
        # Add a small epsilon for numerical stability if total_sum is zero
        cumulative_ratio = cumulative_sum / (total_sum + 1e-8)

        # 4. Find the first index 'k' where the ratio exceeds the threshold.
        # The paper's rank 'k' is 1-based, so the result is index + 1.
        threshold = 1.0 - tau
        # jnp.argmax returns the index of the first 'True' value.
        rank = jnp.argmax(cumulative_ratio >= threshold) + 1
        return rank.astype(jnp.int32)

    def zero_rank():
        # If there are no singular values, the rank is 0.
        return jnp.array(0, dtype=jnp.int32)

    # Use a conditional to handle the edge case of an empty array.
    return jax.lax.cond(s.size > 0, calculate_rank, zero_rank)

@partial(jax.jit, static_argnames=('taus',))
def _compute_dormancy_stats_for_taus(scores: jnp.ndarray, taus: Tuple[float, ...]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled function to compute dormancy stats for multiple thresholds."""
    total_neurons = jnp.array(scores.size, dtype=jnp.int32)
    taus_array = jnp.array(taus)
    def calculate_metrics():
        is_dormant = scores[:, jnp.newaxis] <= taus_array[jnp.newaxis, :]
        dormant_counts = jnp.sum(is_dormant, axis=0)
        dormancy_ratios = dormant_counts / total_neurons
        return dormant_counts.astype(jnp.int32), dormancy_ratios
    def empty_metrics():
        zeros = jnp.zeros_like(taus_array)
        return zeros.astype(jnp.int32), zeros.astype(jnp.float32)
    return jax.lax.cond(total_neurons > 0, calculate_metrics, empty_metrics)

def _calculate_normalized_scores(z: jnp.ndarray) -> jnp.ndarray:
    """Computes normalized scores for a single layer's output tensor."""
    if not isinstance(z, jnp.ndarray) or z.ndim < 2:
        return jnp.array([], dtype=jnp.float32)
    magnitudes = jnp.abs(z)
    axes_to_average = tuple(range(z.ndim - 1))
    neuron_activity = jnp.mean(magnitudes, axis=axes_to_average)
    mean_activity = jnp.mean(neuron_activity)
    # stable_mean_activity = jnp.where(mean_activity == 0, 1e-8, mean_activity)
    stable_mean_activity = mean_activity + 1e-8
    return neuron_activity / stable_mean_activity

def compute_detailed_dormancy_metrics(network, params, batch_stats, states: jnp.ndarray, taus: Tuple[float, ...], action_dim: int, intermediates: Optional[Dict] = None) -> Dict[str, jnp.ndarray]:
    """Computes activation-based dormancy metrics."""
    if intermediates is None:
        _, state = network.apply({"params": params, "batch_stats": batch_stats}, states, train=False, capture_intermediates=True, mutable=["intermediates"])
        intermediates = state.get("intermediates", {})
    leaves, _ = tree_flatten_with_path(intermediates)
    conv_dense_scores, layernorm_scores = [], []
    for path, z in leaves:
        if not (isinstance(z, jnp.ndarray) and z.ndim >= 2 and DictKey('__call__') in path):
            continue
        if action_dim and z.shape[-1] == action_dim:
            continue
        path_str = ".".join(p.key for p in path if isinstance(p, DictKey))
        is_layernorm = "LayerNorm" in path_str
        z_processed = jnp.maximum(z, 0) if is_layernorm else z
        scores = _calculate_normalized_scores(z_processed)
        if scores.size > 0:
            (layernorm_scores if is_layernorm else conv_dense_scores).append(scores)
    all_conv_dense = jnp.concatenate(conv_dense_scores) if conv_dense_scores else jnp.array([])
    all_layernorm = jnp.concatenate(layernorm_scores) if layernorm_scores else jnp.array([])
    cd_counts, cd_ratios = _compute_dormancy_stats_for_taus(all_conv_dense, taus)
    ln_counts, ln_ratios = _compute_dormancy_stats_for_taus(all_layernorm, taus)
    metrics = {'conv_dense/total_neurons': jnp.array(all_conv_dense.size, dtype=jnp.int32), 'layernorm_relu/total_neurons': jnp.array(all_layernorm.size, dtype=jnp.int32)}
    for i, tau in enumerate(taus):
        tau_str = f"{tau:.4f}".rstrip('0').rstrip('.')
        metrics[f'conv_dense/dormancy_ratio_{tau_str}'] = cd_ratios[i]
        metrics[f'conv_dense/dormant_count_{tau_str}'] = cd_counts[i]
        metrics[f'layernorm_relu/dormancy_ratio_{tau_str}'] = ln_ratios[i]
        metrics[f'layernorm_relu/dormant_count_{tau_str}'] = ln_counts[i]
    return metrics

def compute_gradient_based_dormancy_metrics(network, params, batch_stats, perturbations, states, actions, targets, taus: Tuple[float, ...], action_dim: int) -> Dict[str, jnp.ndarray]:
    """Computes gradient-based dormancy metrics (GraMa)."""
    def _loss_fn_for_gradients(perturbations_dict):
        q_vals, _ = network.apply({"params": params, "batch_stats": batch_stats, "perturbations": perturbations_dict}, states, train=True, mutable=["batch_stats"])
        chosen_action_qvals = jnp.take_along_axis(q_vals, actions[:, None], axis=-1).squeeze(axis=-1)
        return 0.5 * jnp.square(chosen_action_qvals - targets).mean()
    grad_perturbations = jax.grad(_loss_fn_for_gradients)(perturbations)
    leaves, _ = tree_flatten_with_path(grad_perturbations)
    post_relu_scores, pre_relu_scores = [], []
    for path, grad_values in leaves:
        if not (isinstance(grad_values, jnp.ndarray) and grad_values.ndim >= 2):
            continue
        if action_dim and grad_values.shape[-1] == action_dim:
            continue
        path_str = ".".join(str(p.key) for p in path if isinstance(p, DictKey))
        is_post_relu = 'post_relu' in path_str.lower()
        scores = _calculate_normalized_scores(grad_values)
        if scores.size > 0:
            (post_relu_scores if is_post_relu else pre_relu_scores).append(scores)
    all_post_relu = jnp.concatenate(post_relu_scores) if post_relu_scores else jnp.array([])
    all_pre_relu = jnp.concatenate(pre_relu_scores) if pre_relu_scores else jnp.array([])
    post_counts, post_ratios = _compute_dormancy_stats_for_taus(all_post_relu, taus)
    pre_counts, pre_ratios = _compute_dormancy_stats_for_taus(all_pre_relu, taus)
    metrics = {'post_relu/total_neurons': jnp.array(all_post_relu.size, dtype=jnp.int32), 'pre_relu/total_neurons': jnp.array(all_pre_relu.size, dtype=jnp.int32)}
    for i, tau in enumerate(taus):
        tau_str = f"{tau:.4f}".rstrip('0').rstrip('.')
        metrics[f'post_relu/dormancy_ratio_{tau_str}'] = post_ratios[i]
        metrics[f'post_relu/dormant_count_{tau_str}'] = post_counts[i]
        metrics[f'pre_relu/dormancy_ratio_{tau_str}'] = pre_ratios[i]
        metrics[f'pre_relu/dormant_count_{tau_str}'] = pre_counts[i]
    return metrics

def compute_combined_dormancy_metrics(network, params, batch_stats, perturbations, minibatch, target, config: Dict, action_dim: int) -> Dict[str, jnp.ndarray]:
    """Computes all dormancy and S-Rank metrics for a single batch."""
    states, actions = minibatch.obs, minibatch.action
    _, state = network.apply({"params": params, "batch_stats": batch_stats}, states, train=False, capture_intermediates=True, mutable=["intermediates"])
    intermediates = state.get("intermediates", {})
    
    # Activation Dormancy
    activation_metrics = compute_detailed_dormancy_metrics(network, params, batch_stats, states, tuple(config["ANALYSIS_KWARGS"]["taus"]), action_dim, intermediates)
    
    # Gradient Dormancy
    gradient_metrics = compute_gradient_based_dormancy_metrics(network, params, batch_stats, perturbations, states, actions, target, tuple(config["ANALYSIS_KWARGS"]["grama_taus"]), action_dim)
    
    # S-Rank
    penultimate_feats = None
    leaves, _ = tree_flatten_with_path(intermediates)
    for path, z in leaves:
        path_str = ".".join(p.key for p in path if isinstance(p, DictKey))
        is_penultimate = ("ImpalaBackbone" in path_str or "CNN" in path_str) and isinstance(path[-1], SequenceKey)
        if is_penultimate and isinstance(z, jnp.ndarray) and z.ndim >= 2:
            penultimate_feats = z
            break
    if penultimate_feats is None:
        penultimate_feats = jnp.zeros((states.shape[0], 1), dtype=states.dtype)
    srank = _compute_srank(penultimate_feats, config["ANALYSIS_KWARGS"]["srank_tau"])
    
    # Combine
    combined_metrics = {f'activation_{k}': v for k, v in activation_metrics.items()}
    combined_metrics.update({f'gradient_{k}': v for k, v in gradient_metrics.items()})
    combined_metrics['srank'] = srank
    return combined_metrics

def restructure_metrics_for_logging(metrics_from_scan: dict) -> list[dict]:
    """Unpacks and deeply restructures metrics for logging."""
    num_epochs = jax.tree_util.tree_leaves(metrics_from_scan)[0].shape[0]
    all_epoch_logs = []
    for i in range(num_epochs):
        epoch_log = {}
        epoch_metrics_scalar = {k: v[i] for k, v in metrics_from_scan.items()}
        for key, value in epoch_metrics_scalar.items():
            parts = key.split('/')
            d = epoch_log
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            final_key = parts[-1]
            if 'dormancy_ratio_' in final_key:
                metric_type, threshold = 'ratio', final_key.replace('dormancy_ratio_', '')
                d = d.setdefault(metric_type, {})
                d[threshold] = value
            elif 'dormant_count_' in final_key:
                metric_type, count = 'count', final_key.replace('dormant_count_', '')
                d = d.setdefault(metric_type, {})
                d[count] = value
            else:
                d[final_key] = value
        all_epoch_logs.append(epoch_log)
    return all_epoch_logs

def restructure_single_step_metrics(flat_metrics: dict) -> dict:
    """
    Restructures a single, flat dictionary of scalar metrics into a nested one.
    Designed to be used inside the real-time logging callback.
    """
    nested_log = {}
    for key, value in flat_metrics.items():
        parts = key.split('/')
        d = nested_log
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        
        final_key = parts[-1]
        
        if 'dormancy_ratio_' in final_key:
            metric_type = 'ratio'
            threshold = final_key.replace('dormancy_ratio_', '')
            d = d.setdefault(metric_type, {})
            d[threshold] = value
        elif 'dormant_count_' in final_key:
            metric_type = 'count'
            threshold = final_key.replace('dormant_count_', '')
            d = d.setdefault(metric_type, {})
            d[threshold] = value
        else:
            d[final_key] = value
            
    return nested_log