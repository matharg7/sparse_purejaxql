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

@partial(jax.jit, static_argnames=('apply_fn',))
def compute_ranks_from_features_jax(feats: jnp.ndarray, tau: float) -> jnp.ndarray:
    feature_matrices = feats[jnp.newaxis, ...]
    threshold = 1.0 - tau
    
    svals = jnp.linalg.svd(feature_matrices, compute_uv=False)

    # (1) Effective rank. Roy & Vetterli (2007)
    sval_sum = jnp.sum(svals, axis=1)
    sval_dist = svals / sval_sum[..., jnp.newaxis]
    sval_dist_fixed = jnp.where(sval_dist == 0, jnp.ones_like(sval_dist), sval_dist)
    effective_ranks = jnp.exp(-jnp.sum(sval_dist_fixed * jnp.log(sval_dist_fixed), axis=1))

    # (2) Approximate rank. PCA variance. Yang et al. (2020)
    sval_squares = svals**2
    sval_squares_sum = jnp.sum(sval_squares, axis=1)
    cumsum_squares = jnp.cumsum(sval_squares, axis=1)
    threshold_crossed_pca = cumsum_squares >= (threshold * sval_squares_sum[..., jnp.newaxis])
    approximate_ranks = jnp.sum(~threshold_crossed_pca, axis=-1) + 1
    
    # (3) srank. Weird. Kumar et al. (2020)
    cumsum = jnp.cumsum(svals, axis=1)
    threshold_crossed_srank = cumsum >= threshold * sval_sum[..., jnp.newaxis]
    sranks = jnp.sum(~threshold_crossed_srank, axis=-1) + 1
    
    # (4) Feature rank. Most basic. Lyle et al. (2022)
    n_obs = jnp.array(feature_matrices.shape[1])
    svals_of_normalized = svals / jnp.sqrt(n_obs)
    over_cutoff = svals_of_normalized > tau
    feature_ranks = jnp.sum(over_cutoff, axis=-1)
    
    # (5) JAX/NumPy rank.
    jax_ranks = jnp.linalg.matrix_rank(feature_matrices)

    # --- Logging ---
    singular_values = {
        "lambda_1": svals_of_normalized[..., 0],
        "lambda_N": svals_of_normalized[..., -1],
    }
    
    if svals_of_normalized.shape[-1] > 1:
        singular_values["lambda_2"] = svals_of_normalized[..., 1]

    ranks = {
        "effective_rank_vetterli": effective_ranks,
        "approximate_rank_pca": approximate_ranks,
        "srank_kumar": sranks,
        "feature_rank_lyle": feature_ranks,
        "jax_rank": jax_ranks,
    }
    
    out = {**singular_values, **ranks}
    # Format keys exactly as in the PyTorch function, without averaging
    out = {f"ranks/{k}": v for k, v in out.items()}
    return out
    
    
    
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
    # jax.debug.breakpoint()
    
    return neuron_activity / stable_mean_activity

def compute_detailed_dormancy_metrics(network, params, batch_stats, states: jnp.ndarray, taus: Tuple[float, ...], action_dim: int, intermediates: Optional[Dict] = None) -> Dict[str, jnp.ndarray]:
    """Computes activation-based dormancy metrics."""
    if intermediates is None:
        _, state = network.apply({"params": params, "batch_stats": batch_stats}, states, train=False, capture_intermediates=True, mutable=["intermediates"])
        intermediates = state.get("intermediates", {})
    leaves, _ = tree_flatten_with_path(intermediates)
    conv_dense_scores, layernorm_scores = [], []
    
    target_modules = ("Conv", "Dense", "LayerNorm")
    targeted_modules = []
    for path, z in leaves:
        if not (isinstance(z, jnp.ndarray) and z.ndim >= 2 and DictKey('__call__') in path):
            continue
        if action_dim and z.shape[-1] == action_dim:
            continue
        # path_str = ".".join(p.key for p in path if isinstance(p, DictKey))
        path_keys = [p.key for p in path if hasattr(p, 'key')]
        if len(path_keys) < 2:
            continue
        module_name = path_keys[-2]
        if not any(target in module_name for target in target_modules):
            continue
        is_layernorm = "LayerNorm" in module_name
        
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
        # metrics[f'conv_dense/dormant_count_{tau_str}'] = cd_counts[i]
        metrics[f'layernorm_relu/dormancy_ratio_{tau_str}'] = ln_ratios[i]
        # metrics[f'layernorm_relu/dormant_count_{tau_str}'] = ln_counts[i]
    return metrics

def compute_gradient_based_dormancy_metrics(network, params, batch_stats, perturbations, states, actions, targets, taus: Tuple[float, ...], action_dim: int) -> Dict[str, jnp.ndarray]:
    """Computes gradient-based dormancy metrics (GraMa)."""
    def _loss_fn_for_gradients(perturbations_dict):
        (q_vals, feat), _ = network.apply({"params": params, "batch_stats": batch_stats, "perturbations": perturbations_dict}, states, train=True, mutable=["batch_stats"])
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
    # jax.debug.breakpoint()
    all_post_relu = jnp.concatenate(post_relu_scores) if post_relu_scores else jnp.array([])
    all_pre_relu = jnp.concatenate(pre_relu_scores) if pre_relu_scores else jnp.array([])
    post_counts, post_ratios = _compute_dormancy_stats_for_taus(all_post_relu, taus)
    pre_counts, pre_ratios = _compute_dormancy_stats_for_taus(all_pre_relu, taus)
    metrics = {'post_relu/total_neurons': jnp.array(all_post_relu.size, dtype=jnp.int32), 'pre_relu/total_neurons': jnp.array(all_pre_relu.size, dtype=jnp.int32)}
    for i, tau in enumerate(taus):
        tau_str = f"{tau:.4f}".rstrip('0').rstrip('.')
        metrics[f'post_relu/dormancy_ratio_{tau_str}'] = post_ratios[i]
        # metrics[f'post_relu/dormant_count_{tau_str}'] = post_counts[i]
        metrics[f'pre_relu/dormancy_ratio_{tau_str}'] = pre_ratios[i]
        # metrics[f'pre_relu/dormant_count_{tau_str}'] = pre_counts[i]
    return metrics

def compute_combined_dormancy_metrics(network, params, batch_stats, perturbations, minibatch, target, config: Dict, action_dim: int) -> Dict[str, jnp.ndarray]:
    """Computes all dormancy and S-Rank metrics for a single batch."""
    states, actions = minibatch.obs, minibatch.action
    (_, penultimate_feats),state = network.apply({"params": params, "batch_stats": batch_stats}, states, train=False, capture_intermediates=True, mutable=["intermediates"])
    intermediates = state.get("intermediates", {})
    # Activation Dormancy
    activation_metrics = compute_detailed_dormancy_metrics(network, params, batch_stats, states, tuple(config["ANALYSIS_KWARGS"]["taus"]), action_dim, intermediates)
    
    # Gradient Dormancy
    gradient_metrics = compute_gradient_based_dormancy_metrics(network, params, batch_stats, perturbations, states, actions, target, tuple(config["ANALYSIS_KWARGS"]["grama_taus"]), action_dim)
    
    # S-Rank Metrics
    srank_metrics = compute_ranks_from_features_jax(penultimate_feats, config["ANALYSIS_KWARGS"]["srank_tau"])
    
    # Combine
    combined_metrics = {f'dormant_neurons/{k}': v for k, v in activation_metrics.items()}
    combined_metrics.update({f'grama/{k}': v for k, v in gradient_metrics.items()})
    combined_metrics.update(srank_metrics)
    
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

# def restructure_single_step_metrics(flat_metrics: dict) -> dict:
#     """
#     Restructures a single, flat dictionary of scalar metrics into a nested one.
#     Designed to be used inside the real-time logging callback.
#     """
#     nested_log = {}
#     for key, value in flat_metrics.items():
#         parts = key.split('/')
#         d = nested_log
#         jax.debug.breakpoint()
#         for part in parts[:-1]:
#             d = d.setdefault(part, {})
        
#         final_key = parts[-1]
        
#         if 'dormancy_ratio_' in final_key:
#             metric_type = 'ratio'
#             threshold = final_key.replace('dormancy_ratio_', '')
#             d = d.setdefault(metric_type, {})
#             d[threshold] = value
#         elif 'dormant_count_' in final_key:
#             metric_type = 'count'
#             threshold = final_key.replace('dormant_count_', '')
#             d = d.setdefault(metric_type, {})
#             d[threshold] = value
#         else:
#             d[final_key] = value 
#     return nested_log

def restructure_single_step_metrics(flat_metrics: dict) -> dict:
    """
    Restructures a single, flat dictionary of scalar metrics into a 
    simpler, flatter dictionary suitable for direct logging.
    
    Example transformations:
    - 'dormant_neurons/conv_dense/dormancy_ratio_0.01' -> 'dormant_neurons/conv_dense/0.01'
    - 'ranks/srank_kumar' -> 'ranks/srank_kumar'
    """
    restructured_log = {}
    for key, value in flat_metrics.items():
        # For 'ranks', the keys are already in the desired format.
        if key.startswith('ranks/'):
            restructured_log[key] = value
            
        # For dormancy and grama metrics, we simplify the key.
        elif key.startswith(('dormant_neurons/', 'grama/')):
            parts = key.split('/')
            final_part = parts[-1]
            
            if 'dormancy_ratio_' in final_part:
                # Extract the threshold value (e.g., '0.01')
                threshold = final_part.replace('dormancy_ratio_', '')
                
                # Reconstruct the key with the main category, sub-type, and threshold
                # e.g., 'dormant_neurons/conv_dense/0.01'
                new_key = f"{parts[0]}/{parts[1]}/{threshold}"
                restructured_log[new_key] = value
            else:
                # This handles other keys like 'total_neurons' which should remain unchanged.
                restructured_log[key] = value
        else:
            # Fallback for any other unexpected keys
            restructured_log[key] = value
            
    return restructured_log