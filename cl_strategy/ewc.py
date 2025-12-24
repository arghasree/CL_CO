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


def train_cl_ewc(train_config, model, train_loader, val_loader, org_weights):

    print("Keys in train_loader:", train_loader.keys())
    print("Keys in val_loader:", val_loader.keys())
    print("Keys in org_weights:", org_weights.keys())
    num_epochs = train_config['num_epochs']
    loss_fn = train_config['loss_fn']
    optimizer = train_config['optimizer']
    rank = train_config['rank']

    ewc_lambda = 1000  # Hyperparameter to scale EWC penalty

    loss = {}
    val_cai = {}
    val_cai_gt = {}

    for org_idx, org in enumerate(train_config['organism']):
        print(f"Training on organism: {org} (Task {org_idx + 1}/{len(train_config['organism'])})")
        loss[org] = []
        val_cai[org] = []
        val_cai_gt[org] = []
        
        if org_idx == 0:
            for epoch in range(num_epochs):
                loss[org].append(normal_train(model, optimizer, train_loader[org], loss_fn, rank, org_weights[org])[0])
                _, val_cai_epoch, val_cai_gt_epoch = validate(model, val_loader[org], loss_fn, org_weights[org], rank, epoch, num_epochs)
                val_cai[org].append(val_cai_epoch)
                val_cai_gt[org].append(val_cai_gt_epoch)
        else:
            old_task = []
            for sub_task in range(org_idx):
                sampled_prev = list(train_loader[train_config['organism'][sub_task]].dataset.get_sample(sample_size=200))
                # print(type(sampled_prev))
                # old_task = old_task + train_loader[train_config['organism'][sub_task]].dataset.get_sample(sample_size=200)
                old_task += list(zip(sampled_prev[0], sampled_prev[1], sampled_prev[2]))
            old_task = random.sample(old_task, 200)
            for epoch in range(num_epochs):
                # print("Type of train_loader[org]: ", type(train_loader[org]))
                # print("Type of org_weights[org]: ", type(org_weights[org]))
                avg_train_loss, _, _ = ewc_train(model, optimizer, train_loader[org], EWC(model, old_task), ewc_lambda, rank, loss_fn, org_weights[org])
                loss[org].append(avg_train_loss)
                _, val_cai_epoch, val_cai_gt_epoch = validate(model, val_loader[org], loss_fn, org_weights[org], rank, epoch, num_epochs)
                val_cai[org].append(val_cai_epoch)
                val_cai_gt[org].append(val_cai_gt_epoch)
        
    return loss, val_cai, val_cai_gt






