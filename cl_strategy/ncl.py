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


def naive_train(model, optimizer, train_loader, loss_fn, rank, org_weights):
    """
    Train model on a single organism (naive fine-tuning).
    This is the standard training without any continual learning regularization.
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        train_loader: Training data loader for current organism
        loss_fn: Loss function
        rank: Device to train on
        org_weights: Codon weights for current organism
    
    Returns:
        avg_train_loss: Average training loss
        avg_train_cai: Average training CAI
        avg_train_cai_gt: Average ground truth CAI
    """
    model.train()
    train_loss = 0
    train_cai = 0
    train_cai_gt = 0

    for i, batch in enumerate(tqdm(train_loader, desc="Training")):
        aa_data = batch['input_ids']
        cds_data = batch['labels']
        organism_id = batch['organism_id']
        
        seq_lens = torch.sum(cds_data != -100, dim=1)
        seq_lens, sorted_index = torch.sort(seq_lens, descending=True)  
        max_seq_len = max(seq_lens)
        
        # Sort by sequence length
        aa_data_sorted = []
        cds_data_sorted = []
        for idx in range(len(sorted_index)):
            aa_data_sorted.append(aa_data[sorted_index[idx]])
            cds_data_sorted.append(cds_data[sorted_index[idx]])
        
        aa_data_sorted = torch.stack(aa_data_sorted)
        cds_data_sorted = torch.stack(cds_data_sorted)
        
        aa_data_sorted = aa_data_sorted.to(rank)
        cds_data_sorted = cds_data_sorted.to(rank)
        organism_id = organism_id.to(rank)
        
        # Forward pass
        output_seq_logits = model(aa_data_sorted, organism_id, seq_lens)
        
        # Trim padding from cds data to match output dimensions
        cds_pad_trimmed = get_pad_trimmed_cds_data(cds_data_sorted, max_seq_len) 
        
        loss = loss_fn(output_seq_logits.permute(0, 2, 1), cds_pad_trimmed.to(rank))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        batch_cai_pred, batch_cai_gt = get_batch_cai(output_seq_logits, cds_data_sorted, seq_lens, org_weights)
        train_cai += torch.mean(batch_cai_pred)
        train_cai_gt += torch.mean(batch_cai_gt)
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_cai = train_cai / len(train_loader)
    avg_train_cai_gt = train_cai_gt / len(train_loader)
    
    return avg_train_loss, avg_train_cai, avg_train_cai_gt


def evaluate_all_tasks(model, val_loaders, org_weights, loss_fn, rank, organisms):
    """
    Evaluate model on all tasks seen so far.
    
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
    
    print(f"  Evaluating on {len(organisms)} task(s)...")
    
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


def calculate_cl_metrics(eval_matrix, organisms):
    """
    Calculate standard continual learning metrics from evaluation matrix.
    
    Metrics:
    1. Average CAI (ACC): Average performance across all tasks after training all
    2. Forgetting: Average amount of performance degradation on previous tasks
    3. Backward Transfer (BWT): Change in performance on previous tasks after learning new ones
    4. Learning Accuracy: Average performance on each task right after training it
    5. Plasticity: Ability to learn new tasks (average initial performance on new tasks)
    
    Args:
        eval_matrix: Dictionary containing evaluation results
        organisms: List of organism names in order
    
    Returns:
        Tuple of (metrics_dict, numpy_matrix_cai, numpy_matrix_cai_gt)
    """
    T = len(organisms)
    R = np.zeros((T, T))  # Evaluation matrix for CAI
    R_gt = np.zeros((T, T))  # Evaluation matrix for CAI_GT
    
    # Fill the evaluation matrix
    for i, train_org in enumerate(organisms):
        for j, eval_org in enumerate(organisms[:i+1]):
            R[i, j] = eval_matrix['cai'][train_org][eval_org]
            R_gt[i, j] = eval_matrix['cai_gt'][train_org][eval_org]
    
    metrics = {}
    
    # 1. Average CAI (final performance on all tasks)
    metrics['avg_cai_final'] = float(np.mean(R[-1, :]))
    metrics['avg_cai_gt_final'] = float(np.mean(R_gt[-1, :]))
    
    # 2. Forgetting (how much performance dropped on old tasks)
    # F_j = max_i(R[i,j]) - R[T,j] for j < T
    forgetting = []
    forgetting_gt = []
    for j in range(T - 1):  # Exclude last task (no forgetting yet)
        max_perf = np.max(R[j:, j])  # Best performance on task j
        final_perf = R[-1, j]  # Final performance on task j
        forgetting.append(max_perf - final_perf)
        
        max_perf_gt = np.max(R_gt[j:, j])
        final_perf_gt = R_gt[-1, j]
        forgetting_gt.append(max_perf_gt - final_perf_gt)
    
    metrics['forgetting'] = float(np.mean(forgetting)) if forgetting else 0.0
    metrics['forgetting_gt'] = float(np.mean(forgetting_gt)) if forgetting_gt else 0.0
    
    # 3. Backward Transfer (impact of learning new tasks on old tasks)
    # BWT = (1/(T-1)) * sum_{i=1}^{T-1} (R[T,i] - R[i,i])
    # Negative BWT indicates catastrophic forgetting
    bwt = []
    bwt_gt = []
    for i in range(T - 1):
        bwt.append(R[-1, i] - R[i, i])
        bwt_gt.append(R_gt[-1, i] - R_gt[i, i])
    
    metrics['backward_transfer'] = float(np.mean(bwt)) if bwt else 0.0
    metrics['backward_transfer_gt'] = float(np.mean(bwt_gt)) if bwt_gt else 0.0
    
    # 4. Learning Accuracy / Plasticity (average diagonal - how well each task was learned initially)
    metrics['learning_accuracy'] = float(np.mean(np.diag(R)))
    metrics['learning_accuracy_gt'] = float(np.mean(np.diag(R_gt)))
    
    # 5. Plasticity: Average performance on each task when first trained
    # This measures the model's ability to learn new tasks
    metrics['plasticity'] = float(np.mean(np.diag(R)))
    metrics['plasticity_gt'] = float(np.mean(np.diag(R_gt)))
    
    # 6. Stability: Inverse of forgetting (how well old knowledge is retained)
    metrics['stability'] = float(1.0 - metrics['forgetting']) if T > 1 else 1.0
    metrics['stability_gt'] = float(1.0 - metrics['forgetting_gt']) if T > 1 else 1.0
    
    # 7. Final performance on each task (for detailed analysis)
    metrics['final_per_task'] = {org: float(R[-1, i]) for i, org in enumerate(organisms)}
    metrics['final_per_task_gt'] = {org: float(R_gt[-1, i]) for i, org in enumerate(organisms)}
    
    # 8. Initial performance on each task (plasticity per task)
    metrics['initial_per_task'] = {org: float(R[i, i]) for i, org in enumerate(organisms)}
    metrics['initial_per_task_gt'] = {org: float(R_gt[i, i]) for i, org in enumerate(organisms)}
    
    return metrics, R, R_gt


def print_cl_metrics(metrics, R, R_gt, organisms):
    """Pretty print continual learning metrics"""
    print("\n" + "="*80)
    print("CONTINUAL LEARNING EVALUATION METRICS")
    print("="*80)
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"   Average CAI (final):        {metrics['avg_cai_final']:.4f}")
    print(f"   Average CAI_GT (final):     {metrics['avg_cai_gt_final']:.4f}")
    print(f"   Learning Accuracy (CAI):    {metrics['learning_accuracy']:.4f}")
    print(f"   Learning Accuracy (CAI_GT): {metrics['learning_accuracy_gt']:.4f}")
    
    print(f"\n STABILITY (Knowledge Retention):")
    print(f"   Stability (CAI):            {metrics['stability']:.4f} (higher is better)")
    print(f"   Stability (CAI_GT):         {metrics['stability_gt']:.4f} (higher is better)")
    
    print(f"\n PLASTICITY (Learning New Tasks):")
    print(f"   Plasticity (CAI):           {metrics['plasticity']:.4f} (higher is better)")
    print(f"   Plasticity (CAI_GT):        {metrics['plasticity_gt']:.4f} (higher is better)")
    
    print(f"\n CATASTROPHIC FORGETTING:")
    print(f"   Forgetting (CAI):           {metrics['forgetting']:.4f} (lower is better)")
    print(f"   Forgetting (CAI_GT):        {metrics['forgetting_gt']:.4f} (lower is better)")
    
    print(f"\nKNOWLEDGE TRANSFER:")
    print(f"   Backward Transfer (CAI):    {metrics['backward_transfer']:.4f} (higher is better)")
    print(f"   Backward Transfer (CAI_GT): {metrics['backward_transfer_gt']:.4f} (higher is better)")
    
    print(f"\nFINAL PERFORMANCE PER TASK (CAI):")
    for org, perf in metrics['final_per_task'].items():
        initial = metrics['initial_per_task'][org]
        change = perf - initial
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"   {org:20s}: {perf:.4f} (initial: {initial:.4f}, change: {arrow} {abs(change):.4f})")
    
    print("\n" + "-"*80)
    print("EVALUATION MATRIX (CAI)")
    print("Rows: After training task i | Columns: Evaluated on task j")
    print("-"*80)
    
    # Header
    header = "After\\On".ljust(20) + "".join([f"{org[:10]}".ljust(12) for org in organisms])
    print(header)
    print("-"*80)
    
    # Matrix rows
    for i, train_org in enumerate(organisms):
        row = f"{train_org[:18]}".ljust(20)
        for j in range(len(organisms)):
            if j <= i:
                row += f"{R[i,j]:.4f}".ljust(12)
            else:
                row += "---".ljust(12)
        print(row)
    
    print("\n" + "-"*80)
    print("EVALUATION MATRIX (CAI_GT)")
    print("-"*80)
    print(header)
    print("-"*80)
    
    for i, train_org in enumerate(organisms):
        row = f"{train_org[:18]}".ljust(20)
        for j in range(len(organisms)):
            if j <= i:
                row += f"{R_gt[i,j]:.4f}".ljust(12)
            else:
                row += "---".ljust(12)
        print(row)
    
    print("="*80 + "\n")


def save_evaluation_results(eval_matrix, cl_metrics, R, R_gt, organisms, save_path):
    """
    Save all evaluation data for later analysis.
    
    Args:
        eval_matrix: Dictionary containing all evaluation results
        cl_metrics: Dictionary of computed continual learning metrics
        R: Numpy array of evaluation matrix (CAI)
        R_gt: Numpy array of evaluation matrix (CAI_GT)
        organisms: List of organism names
        save_path: Directory path to save results
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save as JSON
    results = {
        'evaluation_matrix': eval_matrix,
        'cl_metrics': cl_metrics,
        'numpy_matrix_cai': R.tolist(),
        'numpy_matrix_cai_gt': R_gt.tolist(),
        'organisms': organisms
    }
    
    with open(f"{save_path}/cl_evaluation.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save numpy arrays
    np.save(f"{save_path}/evaluation_matrix_cai.npy", R)
    np.save(f"{save_path}/evaluation_matrix_cai_gt.npy", R_gt)
    
    # Save a summary text file
    with open(f"{save_path}/cl_metrics_summary.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("CONTINUAL LEARNING EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Average CAI (final):        {cl_metrics['avg_cai_final']:.4f}\n")
        f.write(f"Average CAI_GT (final):     {cl_metrics['avg_cai_gt_final']:.4f}\n")
        f.write(f"Stability (CAI):            {cl_metrics['stability']:.4f}\n")
        f.write(f"Stability (CAI_GT):         {cl_metrics['stability_gt']:.4f}\n")
        f.write(f"Plasticity (CAI):           {cl_metrics['plasticity']:.4f}\n")
        f.write(f"Plasticity (CAI_GT):        {cl_metrics['plasticity_gt']:.4f}\n")
        f.write(f"Forgetting (CAI):           {cl_metrics['forgetting']:.4f}\n")
        f.write(f"Forgetting (CAI_GT):        {cl_metrics['forgetting_gt']:.4f}\n")
        f.write(f"Backward Transfer (CAI):    {cl_metrics['backward_transfer']:.4f}\n")
        f.write(f"Backward Transfer (CAI_GT): {cl_metrics['backward_transfer_gt']:.4f}\n")
        f.write(f"Learning Accuracy (CAI):    {cl_metrics['learning_accuracy']:.4f}\n")
        f.write(f"Learning Accuracy (CAI_GT): {cl_metrics['learning_accuracy_gt']:.4f}\n")
        f.write("\n" + "="*80 + "\n")
    
    print(f"\n Results saved to: {save_path}")


def train_naive_continual_learning(train_config, model, train_loader, val_loader, org_weights, save_path="./results/naive"):
    """
    Train model using naive fine-tuning approach (continual learning without regularization).
    This demonstrates catastrophic forgetting and serves as a baseline/lower bound.
    
    Each organism is trained sequentially, and the model simply fine-tunes on each new task
    without any mechanism to preserve previous knowledge.
    
    Args:
        train_config: Dictionary containing training configuration
        model: Neural network model
        train_loader: Dictionary of training loaders for each organism
        val_loader: Dictionary of validation loaders for each organism
        org_weights: Dictionary of organism-specific weights
        save_path: Path to save evaluation results
    
    Returns:
        loss: Training loss per organism and epoch
        val_cai: Validation CAI per organism and epoch
        val_cai_gt: Validation CAI_GT per organism and epoch
        evaluation_matrix: Complete evaluation matrix across all tasks
        cl_metrics: Computed continual learning metrics
    """
    print("\n" + "="*80)
    print("NAIVE CONTINUAL LEARNING (Fine-tuning without Regularization)")
    print("Baseline demonstrating catastrophic forgetting")
    print("="*80)
    
    print("Keys in train_loader:", train_loader.keys())
    print("Keys in val_loader:", val_loader.keys())
    print("Keys in org_weights:", org_weights.keys())
    
    num_epochs = train_config['num_epochs']
    loss_fn = train_config['loss_fn']
    optimizer = train_config['optimizer']
    rank = train_config['rank']
    organisms = train_config['organism']

    loss = {}
    val_cai = {}
    val_cai_gt = {}
    
    # Initialize evaluation matrix
    evaluation_matrix = {
        'cai': {},      # R[i][j] = CAI on task j after training task i
        'cai_gt': {}    # R_gt[i][j] = CAI_GT on task j after training task i
    }

    for org_idx, org in enumerate(organisms):
        print("\n" + "="*80)
        print(f"Training on organism: {org} (Task {org_idx + 1}/{len(organisms)})")
        print("="*80)
        
        loss[org] = []
        val_cai[org] = []
        val_cai_gt[org] = []
        
        # Train on current organism
        print(f"Fine-tuning model on {org}...")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Naive training (just standard fine-tuning, no regularization)
            avg_loss, avg_cai, avg_cai_gt = naive_train(
                model, optimizer, train_loader[org], loss_fn, rank, org_weights[org]
            )
            loss[org].append(avg_loss)
            
            # Validate on current organism
            _, val_cai_epoch, val_cai_gt_epoch = validate(
                model, val_loader[org], loss_fn, org_weights[org], rank, epoch, num_epochs
            )
            val_cai[org].append(val_cai_epoch)
            val_cai_gt[org].append(val_cai_gt_epoch)
            
            print(f"  Train Loss: {avg_loss:.4f}, Val CAI: {val_cai_epoch:.4f}")
        
        # CRITICAL: After training on current task, evaluate on ALL tasks seen so far
        print(f"\n{'='*80}")
        print(f"Evaluating on all {org_idx + 1} task(s) after training {org}...")
        print(f"{'='*80}")
        
        tasks_seen = organisms[:org_idx + 1]
        
        all_task_results = evaluate_all_tasks(
            model, val_loader, org_weights, loss_fn, rank, tasks_seen
        )
        
        # Store in evaluation matrix
        evaluation_matrix['cai'][org] = {}
        evaluation_matrix['cai_gt'][org] = {}
        
        for eval_org in tasks_seen:
            evaluation_matrix['cai'][org][eval_org] = all_task_results[eval_org]['cai']
            evaluation_matrix['cai_gt'][org][eval_org] = all_task_results[eval_org]['cai_gt']
        
        # Show forgetting if not first task
        if org_idx > 0:
            print(f"\n FORGETTING ANALYSIS after training {org}:")
            for prev_org in organisms[:org_idx]:
                current_perf = evaluation_matrix['cai'][org][prev_org]
                initial_perf = evaluation_matrix['cai'][prev_org][prev_org]
                forgetting = initial_perf - current_perf
                print(f"   {prev_org:20s}: {initial_perf:.4f} → {current_perf:.4f} (forgot: {forgetting:.4f})")
    
    # Calculate continual learning metrics after all tasks are complete
    print("\n" + "="*80)
    print("CALCULATING CONTINUAL LEARNING METRICS")
    print("="*80)
    
    cl_metrics, R, R_gt = calculate_cl_metrics(evaluation_matrix, organisms)
    
    # Print metrics
    print_cl_metrics(cl_metrics, R, R_gt, organisms)
    
    # Save results
    save_evaluation_results(evaluation_matrix, cl_metrics, R, R_gt, organisms, save_path)
    
    return loss, val_cai, val_cai_gt, evaluation_matrix, cl_metrics