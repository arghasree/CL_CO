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

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda(device='cuda:1')
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset
        # print("Length of dataset for EWC:", len(self.dataset))
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        torch.backends.cudnn.enabled = False
        for aa_data, cds_data, organism_id in self.dataset:
            # print("AA Data:", aa_data)
            # print("CDS Data:", cds_data)
            # print("Organism ID:", organism_id)
            aa_data = variable(aa_data.unsqueeze(0), use_cuda=False)
            cds_data = variable(cds_data.unsqueeze(0), use_cuda=False)
            organism_id = variable(organism_id.unsqueeze(0), use_cuda=False)
            self.model.zero_grad()
            # print(aa_data.shape, " --- aa_data shape --- ")
            # print(cds_data.shape, " --- cds_data shape --- ")
            # print(organism_id.shape, " --- organism_id shape --- ")

            # print(f'cds data: {cds_data}')
            seq_lens = torch.sum(cds_data != -100, dim=1)
            # seq_lens, sorted_index = torch.sort(seq_lens, descending=True)  
            max_seq_len = max(seq_lens)

            aa_data_sorted = []
            cds_data_sorted = []
            aa_data_sorted.append(aa_data[0])
            cds_data_sorted.append(cds_data[0])
            # for i in range(0, len(sorted_index)):
            #     aa_data_sorted.append(aa_data[sorted_index[i]])
            #     cds_data_sorted.append(cds_data[sorted_index[i]])
            
            aa_data_sorted = torch.stack(aa_data_sorted)
            cds_data_sorted = torch.stack(cds_data_sorted)

            aa_data_sorted = aa_data_sorted.to(device='cuda:1')
            cds_data_sorted = cds_data_sorted.to(device='cuda:1') # sorted sequences by cds length
            organism_id = organism_id.to(device='cuda:1')
            # print the device of seq_lens
            # print("Device of seq_lens:", seq_lens.device)
            output_seq_logits = self.model(aa_data_sorted, organism_id, seq_lens)
           
            cds_pad_trimmed = get_pad_trimmed_cds_data(cds_data_sorted, max_seq_len) 
            # print(output_seq_logits.permute(0,2,1).shape, cds_pad_trimmed.to(rank).shape)
            loss = F.cross_entropy(output_seq_logits.permute(0,2,1), cds_pad_trimmed)

            # output = self.model(input).view(1, -1)
            # label = output.max(1)[1].view(-1)
            # loss = F.cross_entropy(output, label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

def sort_batch_by_length(batch):
    aa_data = batch[0]
    cds_data = batch[1]
    
    seq_lens = torch.sum(cds_data != -100, dim=1)
    seq_lens, sorted_index = torch.sort(seq_lens, descending=True)
    seq_lens = seq_lens.cpu().to(torch.int64)

    aa_data_sorted = []
    cds_data_sorted = []
    for i in range(0, len(sorted_index)):
        aa_data_sorted.append(aa_data[sorted_index[i]])
        cds_data_sorted.append(cds_data[sorted_index[i]])
    
    aa_data_sorted = torch.stack(aa_data_sorted)
    cds_data_sorted = torch.stack(cds_data_sorted)

    return aa_data_sorted, cds_data_sorted, seq_lens
            

def get_max_seq_len(cds_data):
    seq_lens = torch.sum(cds_data != -100, dim=1)
    return max(seq_lens)


def get_pad_trimmed_cds_data(cds_data,  max_seq_len):
    # Trim till max_len 
    # Trims padded portion 
    
    # cds_data_trimmed = [seq[0:max_seq_len] for seq in cds_data]
    # cds_data_trimmed = torch.stack(cds_data_trimmed)
    
    # return cds_data_trimmed
    cds_data_trimmed = []
    for id, seq in enumerate(cds_data):
        # print(type(seq))
        cds_data_trimmed.append(seq[0:max_seq_len])
    
    cds_data_trimmed = torch.stack(cds_data_trimmed)
    return cds_data_trimmed


def normal_train(model, optimizer, train_loader, loss_fn, rank, org_weights):
    model.train()
    train_loss = 0
    train_cai = 0
    train_cai_gt = 0

    for i, batch in enumerate(tqdm(train_loader)):
        aa_data = batch['input_ids']
        cds_data = batch['labels']
        organism_id = batch['organism_id']

        # aa_data = aa_data.to(rank)
        # cds_data = cds_data.to(rank)
        
        seq_lens = torch.sum(cds_data != -100, dim=1)
        seq_lens, sorted_index = torch.sort(seq_lens, descending=True)  
        max_seq_len = max(seq_lens)
        
        # aa_data_sorted, cds_data_sorted, seq_lens = sort_batch_by_length((aa_data, cds_data))
        aa_data_sorted = []
        cds_data_sorted = []
        for i in range(0, len(sorted_index)):
            aa_data_sorted.append(aa_data[sorted_index[i]])
            cds_data_sorted.append(cds_data[sorted_index[i]])
        
        # aa_data = sorted(aa_data, key=lambda x: , reverse=True)
        aa_data_sorted = torch.stack(aa_data_sorted)
        # print("AA DATA", type(aa_data_sorted))
        
        # cds_data = sorted(cds_data, key=lambda x: x.shape[0], reverse=True)
        cds_data_sorted = torch.stack(cds_data_sorted)
        # print("CDS DATA", type(cds_data_sorted))
        

        # print("AA DATA", aa_data_sorted)
        # print("CDS DATA", cds_data_sorted)
        aa_data_sorted = aa_data_sorted.to(rank)
        cds_data_sorted = cds_data_sorted.to(rank) # sorted sequences by cds length
        organism_id = organism_id.to(rank)
        

        #forward
        output_seq_logits = model(aa_data_sorted, organism_id, seq_lens)

        """
        Note the output from pad sequence will only contain
        padded sequences till maxm seq length in the current training batch
        So the dimensions error will come if we try to calculate
        loss with output_seq_logits and cds_data.
        Pack pad sequence takes seq length as input and packs and
        when padding packed sequence the pads are added till maxm seq length
        So now the alternative is to remove the extra pads from the cds_data 
        so that it matches the output_seq_logits dimensions
        """
        # Trim padding from cds data to match output_seq_logits dimensions 
        # as it is packed padded so containd max len as max seq len in current batch
        cds_pad_trimmed = get_pad_trimmed_cds_data(cds_data_sorted, max_seq_len) 

        loss = loss_fn(output_seq_logits.permute(0,2,1), cds_pad_trimmed.to(rank))
        # print("Batch CE Loss: ", loss.item())

        optimizer.zero_grad()
        loss.backward()
        # gradient descent
        optimizer.step()

        # train_batch_count += 1
        train_loss += loss.item()

        batch_cai_pred, batch_cai_gt = get_batch_cai(output_seq_logits, cds_data_sorted, seq_lens, org_weights)
        train_cai += torch.mean(batch_cai_pred)
        train_cai_gt += torch.mean(batch_cai_gt)
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_cai = train_cai / len(train_loader)
    avg_train_cai_gt = train_cai_gt / len(train_loader)
    # train_losses_epoch.append(avg_train_loss)
    return avg_train_loss, avg_train_cai, avg_train_cai_gt

def ewc_train(model, optimizer, train_loader, ewc, ewc_lambda, rank, loss_fn, org_weights):
    model.train()
    train_loss = 0
    train_cai = 0
    train_cai_gt = 0

    for i, batch in enumerate(tqdm(train_loader)):
        aa_data = batch['input_ids']
        cds_data = batch['labels']
        organism_id = batch['organism_id']

        # aa_data = aa_data.to(rank)
        # cds_data = cds_data.to(rank)
        
        seq_lens = torch.sum(cds_data != -100, dim=1)
        seq_lens, sorted_index = torch.sort(seq_lens, descending=True)  
        max_seq_len = max(seq_lens)
        
        # aa_data_sorted, cds_data_sorted, seq_lens = sort_batch_by_length((aa_data, cds_data))
        aa_data_sorted = []
        cds_data_sorted = []
        for i in range(0, len(sorted_index)):
            aa_data_sorted.append(aa_data[sorted_index[i]])
            cds_data_sorted.append(cds_data[sorted_index[i]])
        
        # aa_data = sorted(aa_data, key=lambda x: , reverse=True)
        aa_data_sorted = torch.stack(aa_data_sorted)
        # print("AA DATA", type(aa_data_sorted))
        
        # cds_data = sorted(cds_data, key=lambda x: x.shape[0], reverse=True)
        cds_data_sorted = torch.stack(cds_data_sorted)
        # print("CDS DATA", type(cds_data_sorted))
        

        # print("AA DATA", aa_data_sorted)
        # print("CDS DATA", cds_data_sorted)
        aa_data_sorted = aa_data_sorted.to(rank)
        cds_data_sorted = cds_data_sorted.to(rank) # sorted sequences by cds length
        organism_id = organism_id.to(rank)
        

        #forward
        output_seq_logits = model(aa_data_sorted, organism_id, seq_lens)

        """
        Note the output from pad sequence will only contain
        padded sequences till maxm seq length in the current training batch
        So the dimensions error will come if we try to calculate
        loss with output_seq_logits and cds_data.
        Pack pad sequence takes seq length as input and packs and
        when padding packed sequence the pads are added till maxm seq length
        So now the alternative is to remove the extra pads from the cds_data 
        so that it matches the output_seq_logits dimensions
        """
        # Trim padding from cds data to match output_seq_logits dimensions 
        # as it is packed padded so containd max len as max seq len in current batch
        cds_pad_trimmed = get_pad_trimmed_cds_data(cds_data_sorted, max_seq_len) 

        loss = loss_fn(output_seq_logits.permute(0,2,1), cds_pad_trimmed.to(rank)) + ewc_lambda * ewc.penalty(model)
        # print("Batch CE Loss: ", loss.item())

        optimizer.zero_grad()
        loss.backward()
        # gradient descent
        optimizer.step()

        # train_batch_count += 1
        train_loss += loss.item()

        batch_cai_pred, batch_cai_gt = get_batch_cai(output_seq_logits, cds_data_sorted, seq_lens, org_weights)
        train_cai += torch.mean(batch_cai_pred)
        train_cai_gt += torch.mean(batch_cai_gt)
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_cai = train_cai / len(train_loader)
    avg_train_cai_gt = train_cai_gt / len(train_loader)
    # train_losses_epoch.append(avg_train_loss)
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
            # Use validate function from your model
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
    
    Args:
        eval_matrix: Dictionary containing evaluation results
        organisms: List of organism names in order
    
    Returns:
        Tuple of (metrics_dict, numpy_matrix)
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
    bwt = []
    bwt_gt = []
    for i in range(T - 1):
        bwt.append(R[-1, i] - R[i, i])
        bwt_gt.append(R_gt[-1, i] - R_gt[i, i])
    
    metrics['backward_transfer'] = float(np.mean(bwt)) if bwt else 0.0
    metrics['backward_transfer_gt'] = float(np.mean(bwt_gt)) if bwt_gt else 0.0
    
    # 4. Learning Accuracy (average diagonal - how well each task was learned initially)
    metrics['learning_accuracy'] = float(np.mean(np.diag(R)))
    metrics['learning_accuracy_gt'] = float(np.mean(np.diag(R_gt)))
    
    # 5. Final performance on each task (for detailed analysis)
    metrics['final_per_task'] = {org: float(R[-1, i]) for i, org in enumerate(organisms)}
    metrics['final_per_task_gt'] = {org: float(R_gt[-1, i]) for i, org in enumerate(organisms)}
    
    return metrics, R, R_gt


def print_cl_metrics(metrics, R, R_gt, organisms):
    print("\n" + "="*80)
    print("CONTINUAL LEARNING EVALUATION METRICS")
    print("="*80)
    
    print(f"\n OVERALL PERFORMANCE:")
    print(f"   Average CAI (final):        {metrics['avg_cai_final']:.4f}")
    print(f"   Average CAI_GT (final):     {metrics['avg_cai_gt_final']:.4f}")
    print(f"   Learning Accuracy (CAI):    {metrics['learning_accuracy']:.4f}")
    print(f"   Learning Accuracy (CAI_GT): {metrics['learning_accuracy_gt']:.4f}")
    
    print(f"\n CATASTROPHIC FORGETTING:")
    print(f"   Forgetting (CAI):           {metrics['forgetting']:.4f} (lower is better)")
    print(f"   Forgetting (CAI_GT):        {metrics['forgetting_gt']:.4f} (lower is better)")
    
    print(f"\n KNOWLEDGE TRANSFER:")
    print(f"   Backward Transfer (CAI):    {metrics['backward_transfer']:.4f} (higher is better)")
    print(f"   Backward Transfer (CAI_GT): {metrics['backward_transfer_gt']:.4f} (higher is better)")
    
    print(f"\n FINAL PERFORMANCE PER TASK (CAI):")
    for org, perf in metrics['final_per_task'].items():
        print(f"   {org:20s}: {perf:.4f}")
    
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
        f.write(f"Forgetting (CAI):           {cl_metrics['forgetting']:.4f}\n")
        f.write(f"Forgetting (CAI_GT):        {cl_metrics['forgetting_gt']:.4f}\n")
        f.write(f"Backward Transfer (CAI):    {cl_metrics['backward_transfer']:.4f}\n")
        f.write(f"Backward Transfer (CAI_GT): {cl_metrics['backward_transfer_gt']:.4f}\n")
        f.write(f"Learning Accuracy (CAI):    {cl_metrics['learning_accuracy']:.4f}\n")
        f.write(f"Learning Accuracy (CAI_GT): {cl_metrics['learning_accuracy_gt']:.4f}\n")
        f.write("\n" + "="*80 + "\n")
    
    print(f"\n✅ Results saved to: {save_path}")


def train_cl_ewc(train_config, model, train_loader, val_loader, org_weights, save_path="./results"):
    """
    Train model using Elastic Weight Consolidation (EWC) for continual learning.
    
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
    print("Keys in train_loader:", train_loader.keys())
    print("Keys in val_loader:", val_loader.keys())
    print("Keys in org_weights:", org_weights.keys())
    
    num_epochs = train_config['num_epochs']
    loss_fn = train_config['loss_fn']
    optimizer = train_config['optimizer']
    rank = train_config['rank']
    organisms = train_config['organism']

    ewc_lambda = train_config.get('ewc_lambda', 1000)  # Hyperparameter to scale EWC penalty

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
        
        # Train on current task
        if org_idx == 0:
            # First task: normal training without EWC
            print(f"Task 1: Training without EWC (no previous tasks to consolidate)")
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                avg_loss, _, _ = normal_train(model, optimizer, train_loader[org], loss_fn, rank, org_weights[org])
                loss[org].append(avg_loss)
                
                _, val_cai_epoch, val_cai_gt_epoch = validate(
                    model, val_loader[org], loss_fn, org_weights[org], rank, epoch, num_epochs
                )
                val_cai[org].append(val_cai_epoch)
                val_cai_gt[org].append(val_cai_gt_epoch)
        else:
            # Subsequent tasks: EWC training
            print(f"Task {org_idx + 1}: Training with EWC (consolidating {org_idx} previous task(s))")
            
            # Sample data from previous tasks for EWC
            old_task = []
            for sub_task in range(org_idx):
                prev_org = organisms[sub_task]
                sampled_prev = list(train_loader[prev_org].dataset.get_sample(sample_size=200))
                old_task += list(zip(sampled_prev[0], sampled_prev[1], sampled_prev[2]))
            
            # Randomly sample 200 examples from all previous tasks combined
            old_task = random.sample(old_task, min(200, len(old_task)))
            print(f"  Using {len(old_task)} samples from previous tasks for EWC")
            
            # Create EWC object
            ewc_obj = EWC(model, old_task)
            
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                avg_loss, _, _ = ewc_train(
                    model, optimizer, train_loader[org], ewc_obj, 
                    ewc_lambda, rank, loss_fn, org_weights[org]
                )
                loss[org].append(avg_loss)
                
                _, val_cai_epoch, val_cai_gt_epoch = validate(
                    model, val_loader[org], loss_fn, org_weights[org], rank, epoch, num_epochs
                )
                val_cai[org].append(val_cai_epoch)
                val_cai_gt[org].append(val_cai_gt_epoch)
        
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