import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm.auto import tqdm as tqdm

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.CONSTANTS import AMINO_ACID_DICT, CODON_DICT, SYNONYMOUS_CODONS, ORGANISM_DICT, ORGANISMS
from utils.get_metrics import get_batch_cai

# Define the PyTorch model class
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel,self).__init__()
        #embedding dim changed from 64 to 61
        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=128, padding_idx=0) # 64->62->21
        self.bi_gru = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers=2, #changed from 2-8
            dropout=0.3, # changef from 0.5 -> 0.1
            batch_first=True,
            bidirectional=True
        )
        # self.organism_embedding = nn.Linear(, 32) # org_emb_dim=32
        self.organism_embedding = nn.Embedding(num_embeddings=len(ORGANISMS), embedding_dim=32)
        self.fc1 = nn.Linear(2*256 + 32, 128) # 2*256 because of bidirectional
        self.bn1 = nn.BatchNorm1d(128)
        # self.tanh = nn.ReLU()
        #tanh
        self.tanh_1 = nn.LeakyReLU()
        self.tanh_2 = nn.LeakyReLU()
        self.tanh_3 = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, 256) 
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 61)
        self.token_amino = {v: k for k, v in AMINO_ACID_DICT.items()}
        self.token_codon = {v: k for k, v in CODON_DICT.items()}

    def mask_logits(self, logits, seq_lens, amino_seq):
        # mask = torch.full_like(logits, -1e7)  # Create a mask filled with -1e7

        # for i in range(logits.size(0)):
        #     valid_codon_indices = [CODON_DICT[codon.lower()] for aa in amino_seq[i][:seq_lens[i]] for codon in SYNONYMOUS_CODONS[self.token_amino[aa.item()]]]
        #     mask[i, :seq_lens[i], valid_codon_indices] = 0  # Only valid codons get a zero mask

        # return logits + mask  # Add the mask to the logits; invalid logits get reduced to a very low value
        
        batch_size = logits.size(0)
        mask = torch.full_like(logits, -1e9)
        # print(mask[2, 4:5, :])
        # Iterate over each example in the batch
        for batch_idx in range(batch_size):
            # Iterate over each position in the sequence
            for pos_idx in range(seq_lens[batch_idx]):
                # Get the amino acid at the current position
                amino_acid_idx = amino_seq[batch_idx, pos_idx].item()
                amino_acid = self.token_amino[amino_acid_idx]
                # Get the list of valid codon indices for this amino acid
                valid_codons = SYNONYMOUS_CODONS[amino_acid]
                valid_codon_indices = [CODON_DICT[codon.lower()] for codon in valid_codons]
                # print(valid_codon_indices)
                # Set the mask to 0 (unmask) at the positions of valid codons
                mask[batch_idx, pos_idx, valid_codon_indices] = 0

        # print(mask[2, 4:5, :])
        # Apply the mask to the logits
        masked_logits = logits + mask
        return masked_logits

    def forward(self, x, organism_id, seq_lens):
        """
        seq_lens (the actual length of each sequence before padding)
        mask (attention mask)
        """
        # x (input) is the amino_seq 
        amino_seq = x
        x = self.embedding(x) #  performs an embedding lookup, 
        # converting integer token IDs into dense vector representations.
        # from (batch_size, seq_len) to (batch_size, seq_len, embedding_dim)
        packed_emb = pack_padded_sequence(x, seq_lens, batch_first=True)
        # RNNs process padding tokens unnecessarily. Packing eliminates this by 
        # flattening only the non-padded timesteps and storing metadata about sequence boundaries.
        packed_output, _  = self.bi_gru(packed_emb) # discards the hidden state (only the full output is needed here).
        output, _ = pad_packed_sequence(packed_output, batch_first=True) # shape: (batch_size, seq_len, 2*hidden_dim)
        # print("Output shape after GRU:", output.shape)
        
        org_emb = self.organism_embedding(organism_id)  # shape: (batch_size, seq_len, org_emb_dim)
        
        # This was before
        # org_emb = self.organism_embedding(organism_id.unsqueeze(0)) # shape: (batch_size, org_emb_dim)
        
        # print("Organsim emb shape before unsqueeze:", organism_id.shape)
        # print("Organism embedding shape:", org_emb.shape)
        
        org_emb = org_emb.unsqueeze(1).repeat(1, output.size(1), 1)  # shape: (batch_size, seq_len, org_emb_dim)
        # print("Organism emb shape after unsqueeze and repeat:", org_emb.shape)
        output = torch.cat((output, org_emb), dim=-1)
        # output = torch.cat((output, org_emb.unsqueeze(1).repeat(1, output.size(1), 1)), dim=-1)
        # print("Output shape after concatenating organism embedding:", output.shape)
        # print('Shape of input to fc1:', output.contiguous().view(-1, output.shape[2]).shape)
        output= self.fc1(output.contiguous().view(-1, output.shape[2]))
        output= self.bn1(output)
        output= self.tanh_1(output)
        output= self.fc2(output)
        output= self.bn2(output)
        output= self.tanh_2(output)
        output= self.fc3(output)
        output= self.bn3(output)
        output= self.tanh_3(output)
        output= self.fc4(output)
        output= output.view(x.size(0), -1 , 61)
        output= F.softmax(output, dim=-1)
        # masked_logits = self.mask_logits(output, seq_lens, amino_seq=amino_seq)
        # print("Masking Done")
        # return masked_logits

        return output
    
    
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
    
    
def train(train_config, model, train_loader, org_weights, val_loader=None):
    num_epochs = train_config['num_epochs']
    loss_fn = train_config['loss_fn']
    optimizer = train_config['optimizer']
    rank = train_config['rank']
    
    train_losses_epoch=[]
    
    for epoch in range(num_epochs):
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
        train_losses_epoch.append(avg_train_loss)
        
        
        if val_loader is not None:
            validate(model, val_loader, loss_fn, org_weights, rank, epoch, num_epochs)
            

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Training CAI: {avg_train_cai:.6f}, Training CAI GT: {avg_train_cai_gt:.6f}' )
        # print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Training CAI: {avg_train_cai:.4f}, Training CAI GT: {avg_train_cai_gt:.4f}")
        # print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation CAI: {avg_val_cai:.6f}, Validation CAI GT: {avg_val_cai_gt:.6f}' )
    return train_losses_epoch
   
        
def validate(model, val_loader, loss_fn, org_weights, rank, epoch, num_epochs):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_cai = 0
        val_cai_gt = 0
        for j, val_batch in enumerate(val_loader):
            aa_data = val_batch['input_ids']
            cds_data = val_batch['labels']
            organism_id = val_batch['organism_id']
            # print(f'cds data: {cds_data}')
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
            # print(output_seq_logits.permute(0,2,1).shape, cds_pad_trimmed.to(rank).shape)
            loss = loss_fn(output_seq_logits.permute(0,2,1), cds_pad_trimmed.to(rank))
            val_loss += loss.item()

            batch_cai_pred, batch_cai_gt = get_batch_cai(output_seq_logits, cds_data_sorted, seq_lens, org_weights)

            val_cai += torch.mean(batch_cai_pred)
            val_cai_gt += torch.mean(batch_cai_gt)
        
    avg_val_loss = val_loss / len(val_loader)
    avg_val_cai = val_cai / len(val_loader)
    avg_val_cai_gt = val_cai_gt / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Validation CAI: {avg_val_cai:.4f}, Validation CAI GT: {avg_val_cai_gt:.4f}")

