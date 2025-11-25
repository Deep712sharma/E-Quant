import sys
sys.path.append("./mainldm")
sys.path.append("./mainddpm")
sys.path.append('./src/taming-transformers')
sys.path.append('.')

import torch
import numpy as np
import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def load_clustering_results(clustering_dir: str = "./clustering_output"):
clustering_dir = Path(clustering_dir)
    
    assignments_file = clustering_dir / "cluster_assignments.txt"
    if assignments_file.exists():
        cluster_assignments = np.loadtxt(assignments_file, dtype=np.int32)
        logger.info(f"Loaded cluster assignments: {len(cluster_assignments)} weights")
    else:
        raise FileNotFoundError(f"cluster_assignments.txt not found in {clustering_dir}")
    
    data_file = clustering_dir / "clustering_data.npz"
    if data_file.exists():
        clustering_data = np.load(data_file)
        logger.info(f"Loaded clustering data with keys: {clustering_data.files}")
    else:
        raise FileNotFoundError(f"clustering_data.npz not found in {clustering_dir}")
    
    metadata_file = clustering_dir / "clustering_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata: {list(metadata.keys())}")
    else:
        metadata = {}
        logger.warning(f"clustering_metadata.json not found, using empty metadata")
    
    return cluster_assignments, clustering_data, metadata


def calculate_cluster_densities(cluster_assignments: np.ndarray, 
                                weights: np.ndarray,
                                method: str = "count_over_std") -> List[float]:
unique_clusters = np.unique(cluster_assignments)
    densities = []
    
    for cluster_id in sorted(unique_clusters):
        mask = cluster_assignments == cluster_id
        cluster_weights = weights[mask]
        count = len(cluster_weights)
        
        if method == "count_over_std":
            std = np.std(cluster_weights) + 1e-8 
            density = count / std
        elif method == "count_over_range":
            weight_range = np.ptp(cluster_weights) + 1e-8
            density = count / weight_range
        elif method == "count_only":
            density = float(count)
        elif method == "inverse_variance":
            var = np.var(cluster_weights) + 1e-8
            density = count / var
        else:
            raise ValueError(f"Unknown density method: {method}")
        
        densities.append(float(density))
        
    logger.info(f"Calculated {len(densities)} cluster densities using method: {method}")
    return densities


def organize_clusters_by_layer(cluster_assignments: np.ndarray,
                               weight_shapes: Dict[str, tuple],
                               model) -> Dict[str, Dict]:
layer_clusters = {}
    
    layer_info = []
    total_weights = 0
    
    for layer_name, module in model.named_modules():
        if hasattr(module, 'org_weight'):
            weight = module.org_weight.data.cpu().numpy()
            weight_flat = weight.flatten()
            num_weights = len(weight_flat)
            layer_info.append((layer_name, module, weight, weight_flat, num_weights))
            total_weights += num_weights
            logger.info(f"Found layer {layer_name}: {weight.shape} = {num_weights} weights")
    
    logger.info(f"Total model weights: {total_weights}")
    logger.info(f"Cluster assignments length: {len(cluster_assignments)}")
    
    if len(cluster_assignments) != total_weights:
        logger.error(f"DIMENSION MISMATCH!")
        logger.error(f"  Model has {total_weights} weights")
        logger.error(f"  Cluster assignments has {len(cluster_assignments)} elements")
        logger.error(f"  Difference: {abs(total_weights - len(cluster_assignments))} elements")
        
        if len(cluster_assignments) < total_weights:
            logger.warning("Replicating cluster assignments to match model size...")
            repeat_factor = int(np.ceil(total_weights / len(cluster_assignments)))
            cluster_assignments = np.tile(cluster_assignments, repeat_factor)[:total_weights]
            logger.info(f"Adjusted cluster_assignments to {len(cluster_assignments)} elements")
        
        else:
            logger.warning("Truncating cluster assignments to match model size...")
            cluster_assignments = cluster_assignments[:total_weights]
            logger.info(f"Adjusted cluster_assignments to {len(cluster_assignments)} elements")
    
    start_idx = 0
    
    for layer_name, module, weight, weight_flat, num_weights in layer_info:

        end_idx = start_idx + num_weights
        layer_assignments = cluster_assignments[start_idx:end_idx]
        
        logger.info(f"\nProcessing layer: {layer_name}")
        logger.info(f"  Weight shape: {weight.shape}")
        logger.info(f"  Flat weights: {num_weights}")
        logger.info(f"  Assignments: {len(layer_assignments)}")
        logger.info(f"  Index range: [{start_idx}:{end_idx}]")
        
        if len(layer_assignments) != len(weight_flat):
            logger.error(f"  ERROR: Mismatch for {layer_name}!")
            logger.error(f"    layer_assignments: {len(layer_assignments)}")
            logger.error(f"    weight_flat: {len(weight_flat)}")
            start_idx = end_idx
            continue
        densities = calculate_cluster_densities(layer_assignments, weight_flat)
        
        unique_clusters = np.unique(layer_assignments)
        clusters = {}
        for cluster_id in unique_clusters:
            mask = layer_assignments == cluster_id
            clusters[int(cluster_id)] = torch.tensor(np.where(mask)[0])
        
        assignments_tensor = torch.tensor(layer_assignments.reshape(weight.shape))
        
        layer_clusters[layer_name] = {
            'clusters': clusters,
            'densities': densities,
            'assignments': assignments_tensor,
            'num_clusters': len(unique_clusters)
        }
        
        start_idx = end_idx
        logger.info(f"  ✓ Processed: {len(unique_clusters)} clusters, {len(densities)} densities")
    
    logger.info(f"\nSuccessfully organized {len(layer_clusters)} layers")
    return layer_clusters


def assign_bitwidths_to_clusters(densities: List[float]) -> Dict[int, int]:
n = len(densities)
    sorted_indices = np.argsort(densities)[::-1] 
    
    bitwidth_map = {}
    q1 = n // 4
    q2 = n // 2
    q3 = 3 * n // 4
    
    for idx, cluster_id in enumerate(sorted_indices):
        if idx < q1:
            bitwidth = 8
        elif idx < q2:
            bitwidth = 4
        elif idx < q3:
            bitwidth = 2
        else:
            bitwidth = 1
        
        bitwidth_map[int(cluster_id)] = bitwidth
    
    counts = {8: 0, 4: 0, 2: 0, 1: 0}
    for bw in bitwidth_map.values():
        counts[bw] += 1
    
    logger.info(f"Bit-width distribution:")
    logger.info(f"  8-bit: {counts[8]} clusters ({counts[8]/n*100:.1f}%)")
    logger.info(f"  4-bit: {counts[4]} clusters ({counts[4]/n*100:.1f}%)")
    logger.info(f"  2-bit: {counts[2]} clusters ({counts[2]/n*100:.1f}%)")
    logger.info(f"  1-bit: {counts[1]} clusters ({counts[1]/n*100:.1f}%)")
    
    return bitwidth_map


def prepare_cluster_data_for_quantization(clustering_dir: str,
                                         model,
                                         density_method: str = "count_over_std",
                                         save_path: str = None) -> Dict:
logger.info("=" * 60)
    logger.info("Loading existing clustering results...")
    logger.info("=" * 60)
    
    cluster_assignments, clustering_data, metadata = load_clustering_results(clustering_dir)
    
    if 'weights' in clustering_data.files:
        weights = clustering_data['weights']
    elif 'data' in clustering_data.files:
        weights = clustering_data['data']
    else:
        logger.warning("Weights not found in clustering_data.npz, extracting from model")
        all_weights = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                all_weights.append(module.weight.data.cpu().numpy().flatten())
        weights = np.concatenate(all_weights)
    
    logger.info(f"Total weights: {len(weights)}")
    logger.info(f"Total clusters: {len(np.unique(cluster_assignments))}")
    
    logger.info(f"Calculating cluster densities using method: {density_method}")
    densities = calculate_cluster_densities(cluster_assignments, weights, method=density_method)
    
    logger.info("Assigning bit-widths to clusters based on density...")
    bitwidth_map = assign_bitwidths_to_clusters(densities)

    logger.info("Organizing clusters by layer...")
    weight_shapes = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            weight_shapes[name] = module.weight.shape
    
    layer_clusters = organize_clusters_by_layer(cluster_assignments, weight_shapes, model)
    
    for layer_name in layer_clusters:
        layer_densities = layer_clusters[layer_name]['densities']
        layer_bitwidths = assign_bitwidths_to_clusters(layer_densities)
        layer_clusters[layer_name]['bitwidth_map'] = layer_bitwidths
    
    total_weights = len(cluster_assignments)
    weighted_bits = sum(
        bitwidth_map[int(cid)] for cid in cluster_assignments
    )
    avg_bitwidth = weighted_bits / total_weights
    
    logger.info("=" * 60)
    logger.info(f"Overall Average Bitwidth: {avg_bitwidth:.2f} bits")
    logger.info("=" * 60)
    
    result = {
        'layer_clusters': layer_clusters,
        'global_densities': densities,
        'global_bitwidth_map': bitwidth_map,
        'cluster_assignments': cluster_assignments,
        'avg_bitwidth': avg_bitwidth,
        'metadata': metadata
    }
    
    if save_path:
        torch.save(result, save_path)
        logger.info(f"Saved processed cluster data to: {save_path}")
    
    return result


def apply_cluster_quantization_to_layer(module, cluster_info: Dict, 
                                       weight_quant=True, act_quant=False):
from cluster_based_quantization import ClusterBasedQuantizer

    quantizer = ClusterBasedQuantizer(
        cluster_info['clusters'],
        cluster_info['densities']
    )
    
    if weight_quant and hasattr(module, 'weight'):
        original_weight = module.weight.data.clone()
        quantized_weight = quantizer.quantize_weights(
            original_weight,
            cluster_info['assignments']
        )
        module.weight.data = quantized_weight
        
        avg_bits = sum(
            cluster_info['bitwidth_map'][int(cid.item())] 
            for cid in cluster_info['assignments'].flatten()
        ) / cluster_info['assignments'].numel()
        
        logger.info(f"  Layer average bitwidth: {avg_bits:.2f} bits")
    
    return module

def integrate_with_your_code(model, args):
    clustering_dir = "./clustering_output"
    save_path = f"./calibration/processed_clusters_interval{args.replicate_interval}.pth"
    
    cluster_data = prepare_cluster_data_for_quantization(
        clustering_dir=clustering_dir,
        model=model,
        density_method="count_over_std",
        save_path=save_path
    )
    
    return cluster_data


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    print("Script loaded. Use prepare_cluster_data_for_quantization() or integrate_with_your_code()")
    print("\nExample usage:")
    print("  cluster_data = prepare_cluster_data_for_quantization(")
    print("      clustering_dir='./clustering_output',")
    print("      model=your_model,")
    print("      save_path='./calibration/processed_clusters.pth'")
    print("  )")
