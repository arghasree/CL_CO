import sys

# setting path
sys.path.append('../model')
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils.get_metrics import get_batch_cai
from tqdm.auto import tqdm
from model import validate
import random
import numpy as np
import json
import os
from torch.utils.data import ConcatDataset, DataLoader


def get_pad_trimmed_cds_data(cds_data, max_seq_len):
    """
    Trim till max_len 
    Trims padded portion 
    """
    cds_data_trimmed = []
    for id, seq in enumerate(cds_data):
        cds_data_trimmed.append(seq[0:max_seq_len])
    
    cds_data_trimmed = torch.stack(cds_data_trimmed)
    return cds_data_trimmed


def joint_train_epoch(model, optimizer, train_loader, loss_fn, rank, org_weights_dict, organisms_list):
    """
    Train for one epoch on combined dataset from all organisms.
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        train_loader: Combined DataLoader with all organisms
        loss_fn: Loss function
        rank: Device to train on
        org_weights_dict: Dictionary mapping organism names to their codon weights
        organisms_list: List of organism names in order (for mapping org_id)
    
    Returns:
        avg_train_loss: Average training loss
        avg_train_cai_per_org: Average CAI per organism
        avg_train_cai_gt_per_org: Average ground truth CAI per organism
    """
    model.train()
    train_loss = 0
    train_cai_per_org = {}
    train_cai_gt_per_org = {}
    batch_count_per_org = {}
    
    # Initialize counters for each organism
    for org in org_weights_dict.keys():
        train_cai_per_org[org] = 0
        train_cai_gt_per_org[org] = 0
        batch_count_per_org[org] = 0

    for i, batch in enumerate(tqdm(train_loader, desc="Training")):
        aa_data = batch['input_ids']
        cds_data = batch['labels']
        organism_id = batch['organism_id']
        
        # Calculate sequence lengths BEFORE sorting
        seq_lens = torch.sum(cds_data != -100, dim=1)
        seq_lens, sorted_index = torch.sort(seq_lens, descending=True)  
        max_seq_len = max(seq_lens)
        
        # Sort by sequence length - aa_data and cds_data are already tensors
        # Just reorder using index_select
        aa_data_sorted = aa_data[sorted_index]
        cds_data_sorted = cds_data[sorted_index]
        organism_id_sorted = organism_id[sorted_index]
        
        # Move to device
        aa_data_sorted = aa_data_sorted.to(rank)
        cds_data_sorted = cds_data_sorted.to(rank)
        organism_id_sorted = organism_id_sorted.to(rank)
        
        # Forward pass
        output_seq_logits = model(aa_data_sorted, organism_id_sorted, seq_lens)
        
        # Trim padding from cds data
        cds_pad_trimmed = get_pad_trimmed_cds_data(cds_data_sorted, max_seq_len)
        
        # Calculate loss
        loss = loss_fn(output_seq_logits.permute(0, 2, 1), cds_pad_trimmed.to(rank))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Calculate CAI per organism in batch
        unique_org_ids = torch.unique(organism_id_sorted)
        
        for org_id in unique_org_ids:
            org_id_val = org_id.item()
            
            # Map org_id to organism name
            if org_id_val < len(organisms_list):
                org_name = organisms_list[org_id_val]
            else:
                continue
            
            # Get indices for this organism
            org_mask = organism_id_sorted == org_id
            org_indices = torch.where(org_mask)[0]
            
            if len(org_indices) == 0:
                continue
            
            # Get outputs and labels for this organism
            org_outputs = output_seq_logits[org_indices]
            org_cds = cds_data_sorted[org_indices]
            org_seq_lens = seq_lens[org_indices]
            
            # Calculate CAI for this organism
            batch_cai_pred, batch_cai_gt = get_batch_cai(
                org_outputs, org_cds, org_seq_lens, org_weights_dict[org_name]
            )
            
            train_cai_per_org[org_name] += torch.mean(batch_cai_pred).item()
            train_cai_gt_per_org[org_name] += torch.mean(batch_cai_gt).item()
            batch_count_per_org[org_name] += 1
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Average CAI per organism
    avg_train_cai_per_org = {
        org: (train_cai_per_org[org] / batch_count_per_org[org] if batch_count_per_org[org] > 0 else 0)
        for org in org_weights_dict.keys()
    }
    
    avg_train_cai_gt_per_org = {
        org: (train_cai_gt_per_org[org] / batch_count_per_org[org] if batch_count_per_org[org] > 0 else 0)
        for org in org_weights_dict.keys()
    }
    
    return avg_train_loss, avg_train_cai_per_org, avg_train_cai_gt_per_org


def evaluate_all_tasks_joint(model, val_loaders, org_weights, loss_fn, rank, organisms):
    """
    Evaluate model on all organisms separately.
    
    Args:
        model: The neural network model
        val_loaders: Dictionary of validation loaders for each organism
        org_weights: Dictionary of organism-specific weights
        loss_fn: Loss function
        rank: Device to run on
        organisms: List of organism names to evaluate
    
    Returns:
        Dictionary with CAI and CAI_GT for each organism
    """
    results = {}
    model.eval()
    
    print(f"  Evaluating on {len(organisms)} organism(s)...")
    
    with torch.no_grad():
        for org in organisms:
            _, val_cai, val_cai_gt = validate(
                model, val_loaders[org], loss_fn, 
                org_weights[org], rank, epoch=0, num_epochs=1
            )
            results[org] = {
                'cai': val_cai.item() if torch.is_tensor(val_cai) else val_cai,
                'cai_gt': val_cai_gt.item() if torch.is_tensor(val_cai_gt) else val_cai_gt
            }
            print(f"    {org}: CAI={results[org]['cai']:.4f}, CAI_GT={results[org]['cai_gt']:.4f}")
    
    return results


def create_combined_dataloader(train_loaders, batch_size, shuffle=True, num_workers=0):
    """
    Combine multiple dataloaders into a single dataloader.
    Uses custom collate function to handle variable-length sequences.
    
    Args:
        train_loaders: Dictionary of dataloaders {organism_name: dataloader}
        batch_size: Batch size for combined dataloader
        shuffle: Whether to shuffle the combined dataset
        num_workers: Number of worker processes for data loading
    
    Returns:
        Combined DataLoader
    """
    # Extract datasets from dataloaders
    datasets = []
    for org_name, loader in train_loaders.items():
        datasets.append(loader.dataset)
    
    # Combine datasets
    combined_dataset = ConcatDataset(datasets)
    
    print(f"Combined dataset size: {len(combined_dataset)}")
    for i, (org_name, loader) in enumerate(train_loaders.items()):
        print(f"  {org_name}: {len(loader.dataset)} samples")
    
    # Custom collate function to handle variable-length sequences
    def collate_fn(batch):
        """
        Custom collate function that pads sequences to max length in batch.
        Assumes each item in batch is a dict with keys: 'input_ids', 'labels', 'organism_id'
        """
        # Get max lengths in this batch
        max_aa_len = max([item['input_ids'].shape[0] for item in batch])
        max_cds_len = max([item['labels'].shape[0] for item in batch])
        
        # Pad sequences
        input_ids_list = []
        labels_list = []
        organism_ids = []
        
        for item in batch:
            # Pad input_ids (amino acids)
            aa_data = item['input_ids']
            aa_pad_len = max_aa_len - aa_data.shape[0]
            if aa_pad_len > 0:
                aa_padded = torch.cat([aa_data, torch.zeros(aa_pad_len, dtype=aa_data.dtype)])
            else:
                aa_padded = aa_data
            input_ids_list.append(aa_padded)
            
            # Pad labels (codons) with -100
            cds_data = item['labels']
            cds_pad_len = max_cds_len - cds_data.shape[0]
            if cds_pad_len > 0:
                cds_padded = torch.cat([cds_data, torch.full((cds_pad_len,), -100, dtype=cds_data.dtype)])
            else:
                cds_padded = cds_data
            labels_list.append(cds_padded)
            
            # Organism ID
            organism_ids.append(item['organism_id'])
        
        # Stack into tensors
        batch_dict = {
            'input_ids': torch.stack(input_ids_list),
            'labels': torch.stack(labels_list),
            'organism_id': torch.stack(organism_ids)
        }
        
        return batch_dict
    
    # Create combined dataloader with custom collate
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return combined_loader


def calculate_joint_metrics(epoch_results, organisms):
    """
    Calculate metrics for joint training across epochs.
    
    Args:
        epoch_results: List of dictionaries containing results per epoch
        organisms: List of organism names
    
    Returns:
        Dictionary of computed metrics
    """
    num_epochs = len(epoch_results)
    
    metrics = {}
    
    # Final performance on each organism
    final_results = epoch_results[-1]
    metrics['final_cai_per_org'] = final_results['val_cai']
    metrics['final_cai_gt_per_org'] = final_results['val_cai_gt']
    
    # Average across organisms
    metrics['avg_final_cai'] = np.mean(list(final_results['val_cai'].values()))
    metrics['avg_final_cai_gt'] = np.mean(list(final_results['val_cai_gt'].values()))
    
    # Training progression (first vs last epoch)
    first_results = epoch_results[0]
    metrics['improvement_per_org'] = {
        org: final_results['val_cai'][org] - first_results['val_cai'][org]
        for org in organisms
    }
    
    metrics['avg_improvement'] = np.mean(list(metrics['improvement_per_org'].values()))
    
    # Std across organisms (measure of balance)
    metrics['std_final_cai'] = np.std(list(final_results['val_cai'].values()))
    metrics['std_final_cai_gt'] = np.std(list(final_results['val_cai_gt'].values()))
    
    return metrics


def print_joint_training_summary(metrics, organisms, epoch_results):
    """
    Pretty print joint training results.
    """
    print("\n" + "="*80)
    print("JOINT MULTI-TASK TRAINING RESULTS")
    print("="*80)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Average CAI (final):     {metrics['avg_final_cai']:.4f}")
    print(f"   Average CAI_GT (final):  {metrics['avg_final_cai_gt']:.4f}")
    print(f"   Std CAI across organisms: {metrics['std_final_cai']:.4f}")
    print(f"   Average improvement:     {metrics['avg_improvement']:.4f}")
    
    print(f"\n📋 FINAL PERFORMANCE PER ORGANISM:")
    print("-" * 80)
    print(f"{'Organism':<25} {'CAI':>12} {'CAI_GT':>12} {'Improvement':>12}")
    print("-" * 80)
    
    for org in organisms:
        cai = metrics['final_cai_per_org'][org]
        cai_gt = metrics['final_cai_gt_per_org'][org]
        improvement = metrics['improvement_per_org'][org]
        print(f"{org:<25} {cai:>12.4f} {cai_gt:>12.4f} {improvement:>12.4f}")
    
    print("-" * 80)
    
    # # Training curves summary
    # print(f"\n📈 TRAINING PROGRESSION:")
    # print(f"   First epoch avg CAI:  {epoch_results[0]['train_cai_avg']:.4f}")
    # print(f"   Last epoch avg CAI:   {epoch_results[-1]['train_cai_avg']:.4f}")
    # print(f"   First epoch loss:     {epoch_results[0]['train_loss']:.4f}")
    # print(f"   Last epoch loss:      {epoch_results[-1]['train_loss']:.4f}")
    
    # print("="*80 + "\n")


def save_joint_training_results(metrics, epoch_results, organisms, save_path):
    """
    Save joint training results.
    
    Args:
        metrics: Dictionary of computed metrics
        epoch_results: List of results per epoch
        organisms: List of organism names
        save_path: Directory to save results
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save as JSON
    results = {
        'metrics': metrics,
        'epoch_results': epoch_results,
        'organisms': organisms,
        'num_epochs': len(epoch_results)
    }
    
    with open(f"{save_path}/joint_training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary text
    with open(f"{save_path}/joint_training_summary.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("JOINT MULTI-TASK TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Number of organisms: {len(organisms)}\n")
        f.write(f"Organisms: {', '.join(organisms)}\n\n")
        f.write(f"Average final CAI:        {metrics['avg_final_cai']:.4f}\n")
        f.write(f"Average final CAI_GT:     {metrics['avg_final_cai_gt']:.4f}\n")
        f.write(f"Std CAI across organisms: {metrics['std_final_cai']:.4f}\n")
        f.write(f"Average improvement:      {metrics['avg_improvement']:.4f}\n\n")
        
        f.write("Per-organism final performance:\n")
        f.write("-" * 80 + "\n")
        for org in organisms:
            f.write(f"{org:25s}: CAI={metrics['final_cai_per_org'][org]:.4f}, ")
            f.write(f"CAI_GT={metrics['final_cai_gt_per_org'][org]:.4f}\n")
        f.write("="*80 + "\n")
    
    # Save training curves as numpy
    training_losses = [epoch['train_loss'] for epoch in epoch_results]
    np.save(f"{save_path}/training_losses.npy", np.array(training_losses))
    
    # Save per-organism CAI curves
    for org in organisms:
        org_cai = [epoch['val_cai'][org] for epoch in epoch_results]
        org_cai_gt = [epoch['val_cai_gt'][org] for epoch in epoch_results]
        np.save(f"{save_path}/val_cai_{org}.npy", np.array(org_cai))
        np.save(f"{save_path}/val_cai_gt_{org}.npy", np.array(org_cai_gt))
    
    print(f"\n✅ Results saved to: {save_path}")


def train_joint_multitask(train_config, model, train_loader, val_loader, org_weights, save_path="./results/joint"):
    """
    Train model jointly on all organisms simultaneously (multi-task learning).
    This serves as the upper bound for continual learning experiments.
    
    Args:
        train_config: Dictionary containing training configuration
        model: Neural network model
        train_loader: Dictionary of training loaders for each organism
        val_loader: Dictionary of validation loaders for each organism
        org_weights: Dictionary of organism-specific weights
        save_path: Path to save results
    
    Returns:
        epoch_results: List of results per epoch
        metrics: Computed performance metrics
    """
    print("\n" + "="*80)
    print("JOINT MULTI-TASK TRAINING (Upper Bound for Continual Learning)")
    print("="*80)
    
    num_epochs = train_config['num_epochs']
    loss_fn = train_config['loss_fn']
    optimizer = train_config['optimizer']
    rank = train_config['rank']
    organisms = train_config['organism']
    batch_size = train_config.get('batch_size', 32)
    num_workers = train_config.get('num_workers', 0)
    
    print(f"\nTraining configuration:")
    print(f"  Number of organisms: {len(organisms)}")
    print(f"  Organisms: {', '.join(organisms)}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    
    # Create combined dataloader
    print("\nCombining datasets...")
    combined_train_loader = create_combined_dataloader(
        train_loader, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    
    # Track results per epoch
    epoch_results = []
    
    # Training loop
    for epoch in range(num_epochs):
        print("\n" + "="*80)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("="*80)
        
        # Train for one epoch
        train_loss, train_cai_per_org, train_cai_gt_per_org = joint_train_epoch(
            model, optimizer, combined_train_loader, loss_fn, rank, org_weights, organisms
        )
        
        print(f"\n📉 Training Loss: {train_loss:.4f}")
        print(f"📊 Average Training CAI: {np.mean(list(train_cai_per_org.values())):.4f}")
        
        # Evaluate on all organisms separately
        print(f"\n{'='*80}")
        print(f"Evaluating on all {len(organisms)} organism(s)...")
        print(f"{'='*80}")
        
        val_results = evaluate_all_tasks_joint(
            model, val_loader, org_weights, loss_fn, rank, organisms
        )
        
        # Extract CAI values
        val_cai = {org: val_results[org]['cai'] for org in organisms}
        val_cai_gt = {org: val_results[org]['cai_gt'] for org in organisms}
        
        # Store epoch results
        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_cai_per_org': train_cai_per_org,
            'train_cai_gt_per_org': train_cai_gt_per_org,
            'train_cai_avg': np.mean(list(train_cai_per_org.values())),
            'val_cai': val_cai,
            'val_cai_gt': val_cai_gt,
            'val_cai_avg': np.mean(list(val_cai.values())),
            'val_cai_gt_avg': np.mean(list(val_cai_gt.values()))
        })
        
        print(f"\n📊 Validation Summary:")
        print(f"   Average CAI:    {epoch_results[-1]['val_cai_avg']:.4f}")
        print(f"   Average CAI_GT: {epoch_results[-1]['val_cai_gt_avg']:.4f}")
    
    # Calculate final metrics
    print("\n" + "="*80)
    print("CALCULATING FINAL METRICS")
    print("="*80)
    
    metrics = calculate_joint_metrics(epoch_results, organisms)
    
    # Print summary
    print_joint_training_summary(metrics, organisms, epoch_results)
    
    # Save results
    save_joint_training_results(metrics, epoch_results, organisms, save_path)
    
    return epoch_results, metrics


def train_joint_multitask_simple(train_config, model, train_loader, val_loader, org_weights, save_path="./results/joint"):
    """
    Simplified version that treats combined data as single task.
    Useful if you don't need per-organism tracking during training.
    
    Args:
        train_config: Dictionary containing training configuration
        model: Neural network model
        train_loader: Dictionary of training loaders for each organism
        val_loader: Dictionary of validation loaders for each organism
        org_weights: Dictionary of organism-specific weights
        save_path: Path to save results
    
    Returns:
        epoch_results: List of results per epoch
        metrics: Computed performance metrics
    """
    print("\n" + "="*80)
    print("JOINT MULTI-TASK TRAINING (Simplified)")
    print("="*80)
    
    num_epochs = train_config['num_epochs']
    loss_fn = train_config['loss_fn']
    optimizer = train_config['optimizer']
    rank = train_config['rank']
    organisms = train_config['organism']
    batch_size = train_config.get('batch_size', 32)
    num_workers = train_config.get('num_workers', 0)
    
    # Create combined dataloader
    combined_train_loader = create_combined_dataloader(
        train_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    epoch_results = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Simple training (no per-organism tracking)
        model.train()
        train_loss = 0
        
        for batch in tqdm(combined_train_loader, desc="Training"):
            aa_data = batch['input_ids']
            cds_data = batch['labels']
            organism_id = batch['organism_id']
            
            seq_lens = torch.sum(cds_data != -100, dim=1)
            seq_lens, sorted_index = torch.sort(seq_lens, descending=True)
            max_seq_len = max(seq_lens)
            
            # Use index_select instead of stacking individual elements
            aa_data_sorted = aa_data[sorted_index]
            cds_data_sorted = cds_data[sorted_index]
            organism_id_sorted = organism_id[sorted_index]
            
            aa_data_sorted = aa_data_sorted.to(rank)
            cds_data_sorted = cds_data_sorted.to(rank)
            organism_id_sorted = organism_id_sorted.to(rank)
            
            output_seq_logits = model(aa_data_sorted, organism_id_sorted, seq_lens)
            cds_pad_trimmed = get_pad_trimmed_cds_data(cds_data_sorted, max_seq_len)
            
            loss = loss_fn(output_seq_logits.permute(0, 2, 1), cds_pad_trimmed.to(rank))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(combined_train_loader)
        
        # Evaluate on each organism
        # print("Val loader is ", val_loader)
        val_results = evaluate_all_tasks_joint(
            model, val_loader, org_weights, loss_fn, rank, organisms
        )
        
        val_cai = {org: val_results[org]['cai'] for org in organisms}
        val_cai_gt = {org: val_results[org]['cai_gt'] for org in organisms}
        
        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_cai': val_cai,
            'val_cai_gt': val_cai_gt,
            'val_cai_avg': np.mean(list(val_cai.values()))
        })
        
        print(f"Loss: {avg_train_loss:.4f}, Avg CAI: {epoch_results[-1]['val_cai_avg']:.4f}")
    
    metrics = calculate_joint_metrics(epoch_results, organisms)
    print_joint_training_summary(metrics, organisms, epoch_results)
    save_joint_training_results(metrics, epoch_results, organisms, save_path)
    
    return epoch_results, metrics