import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm.auto import tqdm as tqdm
from torch.nn import CrossEntropyLoss


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.CONSTANTS import AMINO_ACID_DICT, CODON_DICT, SYNONYMOUS_CODONS, ORGANISMS
from utils.get_metrics import get_batch_cai
from CL_CO.cl_strategy.ewc import calculate_cl_metrics, print_cl_metrics, save_evaluation_results, evaluate_all_tasks


    
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


def loss_cross_entropy_plus_l2(output, target, old_task_parameters, model, l2_lambda=1e-5):
    """
    Custom loss function that combines Cross Entropy Loss with L2 regularization.
    """
    ce_loss_fn = CrossEntropyLoss()
    ce_loss = ce_loss_fn(output, target)

    l2_reg = 0.0
    for p, old_p in zip(model.parameters(), old_task_parameters):
        l2_reg += torch.sum((p - old_p) ** 2)


    total_loss = ce_loss + l2_lambda * l2_reg
    return total_loss


def normal_loss(output, target):
    ce_loss_fn = CrossEntropyLoss()
    ce_loss = ce_loss_fn(output, target)
    return ce_loss


def normal_train(train_config, model, train_loader, org_weights, val_loader=None):
    num_epochs = train_config['num_epochs']
    optimizer = train_config['optimizer']
    rank = train_config['rank']
    
    train_losses_epoch=[]
    
    # for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_cai = 0
    train_cai_gt = 0

    for i, batch in enumerate(tqdm(train_loader)):
        aa_data = batch['input_ids']
        cds_data = batch['labels']
        organism_id = batch['organism_id']
        
        seq_lens = torch.sum(cds_data != -100, dim=1)
        seq_lens, sorted_index = torch.sort(seq_lens, descending=True)  
        max_seq_len = max(seq_lens)
        
        aa_data_sorted = []
        cds_data_sorted = []
        for i in range(0, len(sorted_index)):
            aa_data_sorted.append(aa_data[sorted_index[i]])
            cds_data_sorted.append(cds_data[sorted_index[i]])
        
        
        aa_data_sorted = torch.stack(aa_data_sorted)
        cds_data_sorted = torch.stack(cds_data_sorted)

        aa_data_sorted = aa_data_sorted.to(rank)
        cds_data_sorted = cds_data_sorted.to(rank) # sorted sequences by cds length
        organism_id = organism_id.to(rank)
        

        #forward
        output_seq_logits = model(aa_data_sorted, organism_id, seq_lens)

        # Trim padding from cds data to match output_seq_logits dimensions 
        # as it is packed padded so containd max len as max seq len in current batch
        cds_pad_trimmed = get_pad_trimmed_cds_data(cds_data_sorted, max_seq_len) 
        
        loss = normal_loss(output_seq_logits.permute(0,2,1), cds_pad_trimmed.to(rank))
        # print("Batch CE Loss: ", loss.item())

        optimizer.zero_grad()
        loss.backward()
        # gradient descent
        optimizer.step()

        # train_batch_count += 1
        # train_loss += loss.item()
    return loss
        

    #     batch_cai_pred, batch_cai_gt = get_batch_cai(output_seq_logits, cds_data_sorted, seq_lens, org_weights)
    #     train_cai += torch.mean(batch_cai_pred)
    #     train_cai_gt += torch.mean(batch_cai_gt)
        
    #     avg_train_loss = train_loss / len(train_loader)
    #     avg_train_cai = train_cai / len(train_loader)
    #     avg_train_cai_gt = train_cai_gt / len(train_loader)
    #     train_losses_epoch.append(avg_train_loss)
        
    #     # if val_loader is not None:
    #     #     validate(model, val_loader, loss_fn, org_weights, rank, epoch, num_epochs)
            

    #     print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Training CAI: {avg_train_cai:.4f}, Training CAI GT: {avg_train_cai_gt:.4f}")
    
    # return train_losses_epoch, avg_train_cai_gt, avg_train_cai
   
    
def train(train_config, model, train_loader, org_weights, val_loader=None):
    num_epochs = train_config['num_epochs']
    loss_fn = train_config['loss_fn']
    optimizer = train_config['optimizer']
    rank = train_config['rank']
    
    train_losses_epoch=[]
    
    # for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_cai = 0
    train_cai_gt = 0

    for i, batch in enumerate(tqdm(train_loader)):
        aa_data = batch['input_ids']
        cds_data = batch['labels']
        organism_id = batch['organism_id']
        
        seq_lens = torch.sum(cds_data != -100, dim=1)
        seq_lens, sorted_index = torch.sort(seq_lens, descending=True)  
        max_seq_len = max(seq_lens)
        
        aa_data_sorted = []
        cds_data_sorted = []
        for i in range(0, len(sorted_index)):
            aa_data_sorted.append(aa_data[sorted_index[i]])
            cds_data_sorted.append(cds_data[sorted_index[i]])
        
        aa_data_sorted = torch.stack(aa_data_sorted)
        cds_data_sorted = torch.stack(cds_data_sorted)

        aa_data_sorted = aa_data_sorted.to(rank)
        cds_data_sorted = cds_data_sorted.to(rank) # sorted sequences by cds length
        organism_id = organism_id.to(rank)
        

        #forward
        output_seq_logits = model(aa_data_sorted, organism_id, seq_lens)

        # Trim padding from cds data to match output_seq_logits dimensions 
        # as it is packed padded so containd max len as max seq len in current batch
        cds_pad_trimmed = get_pad_trimmed_cds_data(cds_data_sorted, max_seq_len) 
        
        old_task_parameters = train_config['old_task_parameters']
        loss = loss_cross_entropy_plus_l2(output_seq_logits.permute(0,2,1), cds_pad_trimmed, old_task_parameters, model)

        optimizer.zero_grad()
        loss.backward()
        # gradient descent
        optimizer.step()
        
    return loss

        # train_batch_count += 1
        # train_loss += loss.item()

    #     batch_cai_pred, batch_cai_gt = get_batch_cai(output_seq_logits, cds_data_sorted, seq_lens, org_weights)
    #     train_cai += torch.mean(batch_cai_pred)
    #     train_cai_gt += torch.mean(batch_cai_gt)
    
    # avg_train_loss = train_loss / len(train_loader)
    # avg_train_cai = train_cai / len(train_loader)
    # avg_train_cai_gt = train_cai_gt / len(train_loader)
    # train_losses_epoch.append(avg_train_loss)
    
    
    # if val_loader is not None:
    #     validate(model, val_loader, org_weights, rank, label='Validation')
        

    # print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Training CAI: {avg_train_cai:.4f}, Training CAI GT: {avg_train_cai_gt:.4f}")
    
    # return train_losses_epoch, avg_train_cai_gt, avg_train_cai
   
        
def validate(model, val_loader, org_weights, rank):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_cai = 0
        val_cai_gt = 0
        for j, val_batch in enumerate(val_loader):
            aa_data = val_batch['input_ids']
            cds_data = val_batch['labels']
            organism_id = val_batch['organism_id']

            seq_lens = torch.sum(cds_data != -100, dim=1)
            seq_lens, sorted_index = torch.sort(seq_lens, descending=True)  
            max_seq_len = max(seq_lens)

            aa_data_sorted = []
            cds_data_sorted = []
            for i in range(0, len(sorted_index)):
                aa_data_sorted.append(aa_data[sorted_index[i]])
                cds_data_sorted.append(cds_data[sorted_index[i]])
            
            aa_data_sorted = torch.stack(aa_data_sorted)
            cds_data_sorted = torch.stack(cds_data_sorted)

            aa_data_sorted = aa_data_sorted.to(rank)
            cds_data_sorted = cds_data_sorted.to(rank)
            organism_id = organism_id.to(rank)

            output_seq_logits = model(aa_data_sorted, organism_id, seq_lens)

            cds_pad_trimmed = get_pad_trimmed_cds_data(cds_data_sorted, max_seq_len) 

            loss = normal_loss(output_seq_logits.permute(0,2,1), cds_pad_trimmed.to(rank))
            val_loss += loss.item()

            batch_cai_pred, batch_cai_gt = get_batch_cai(output_seq_logits, cds_data_sorted, seq_lens, org_weights)

            val_cai += torch.mean(batch_cai_pred)
            val_cai_gt += torch.mean(batch_cai_gt)
        
    avg_val_loss = val_loss / len(val_loader)
    avg_val_cai = val_cai / len(val_loader)
    avg_val_cai_gt = val_cai_gt / len(val_loader)
    
    # print(f"{label} Loss: {avg_val_loss:.4f}, {label} CAI: {avg_val_cai:.4f}, {label} CAI GT: {avg_val_cai_gt:.4f}")
    
    return avg_val_loss, avg_val_cai, avg_val_cai_gt


def train_l2_strategy(train_config, model, train_loader, val_loader, org_weights, save_path="./results"):
    print("Keys in train_loader:", train_loader.keys())
    print("Keys in val_loader:", val_loader.keys())
    print("Keys in org_weights:", org_weights.keys())
    
    num_epochs = train_config['num_epochs']
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
        
        # Train on current task
        if org_idx == 0:
            # First task: normal training without EWC
            print(f"Task 1: Training without EWC (no previous tasks to consolidate)")
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                avg_loss=normal_train(train_config, model, train_loader[org], org_weights[org], val_loader[org])
                loss[org].append(avg_loss)
                
                _, val_cai_epoch, val_cai_gt_epoch = validate(model, val_loader[org], org_weights[org], rank)
                
                val_cai[org].append(val_cai_epoch)
                val_cai_gt[org].append(val_cai_gt_epoch)
        else:
            # Subsequent tasks: EWC training
            print(f"Task {org_idx + 1}")
            old_task_parameters = [p.clone().detach() for p in model.parameters()]
            train_config['old_task_parameters'] = old_task_parameters
            
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                avg_loss = train(train_config, model, train_loader[org], org_weights[org], val_loader[org])
                loss[org].append(avg_loss)
                
                _, val_cai_epoch, val_cai_gt_epoch = validate(model, val_loader[org], org_weights[org], rank)
                
                val_cai[org].append(val_cai_epoch)
                val_cai_gt[org].append(val_cai_gt_epoch)
        
        # CRITICAL: After training on current task, evaluate on ALL tasks seen so far
        print(f"\n{'='*80}")
        print(f"Evaluating on all {org_idx + 1} task(s) after training {org}...")
        print(f"{'='*80}")
        
        tasks_seen = organisms[:org_idx + 1]
        
        all_task_results = evaluate_all_tasks(
            model, val_loader, org_weights, normal_loss, rank, tasks_seen
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