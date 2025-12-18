import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm.auto import tqdm as tqdm

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from CONSTANTS import AMINO_ACID_DICT, CODON_DICT, SYNONYMOUS_CODONS
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

        self.fc1 = nn.Linear(2*256, 128) # 2*256 because of bidirectional
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

    def forward(self, x, seq_lens, mask):
        """
        seq_lens (the actual length of each sequence before padding)
        mask (attention mask)
        """
        # x (input) is the amino_seq 
        x = self.embedding(x) #  performs an embedding lookup, 
        # converting integer token IDs into dense vector representations.
        # from (batch_size, seq_len) to (batch_size, seq_len, embedding_dim)
        packed_emb = pack_padded_sequence(x, seq_lens, batch_first=True)
        # RNNs process padding tokens unnecessarily. Packing eliminates this by 
        # flattening only the non-padded timesteps and storing metadata about sequence boundaries.
        packed_output, _  = self.bi_gru(packed_emb) # discards the hidden state (only the full output is needed here).
        output, _ = pad_packed_sequence(packed_output, batch_first=True) # shape: (batch_size, seq_len, 2*hidden_dim)

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

        return output
    
    
def sort_batch_by_length(batch):
    aa_data = batch[0]
    cds_data = batch[1]
    
    seq_lens = torch.sum(cds_data != -100, dim=1)
    seq_lens, sorted_index = torch.sort(seq_lens, descending=True)

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


def get_pad_trimmed_cds_data(cds_data):
    # Trim till max_len 
    # Trims padded portion 
    
    max_seq_len = get_max_seq_len(cds_data)
    
    cds_data_trimmed = [seq[0:max_seq_len] for seq in cds_data]
    cds_data_trimmed = torch.stack(cds_data_trimmed)
    
    return cds_data_trimmed
    
    
def train(train_config, model, train_loader):
    num_epochs = train_config['num_epochs']
    cross_entropy_loss = train_config['loss_fn']
    optimizer = train_config['optimizer']
    rank = train_config['rank']
    
    train_losses_epoch=[]
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i, batch in enumerate(tqdm(train_loader)):
            aa_data = batch['input_ids']
            cds_data = batch['labels']
    
            aa_data = aa_data.to(rank)
            cds_data = cds_data.to(rank)
            
            aa_data_sorted, cds_data_sorted, seq_lens = sort_batch_by_length((aa_data, cds_data))

            aa_data_sorted = aa_data_sorted.to(rank)
            cds_data_sorted = cds_data_sorted.to(rank) # sorted sequences by cds length

            #forward
            output_seq_logits = model(aa_data_sorted, seq_lens)

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
            cds_pad_trimmed = get_pad_trimmed_cds_data(cds_data_sorted) 

            total_loss = cross_entropy_loss(output_seq_logits.permute(0,2,1), cds_pad_trimmed.to(rank))
            # print("Batch CE Loss: ", loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            # gradient descent
            optimizer.step()

            # train_batch_count += 1
            train_loss += total_loss.item()


        avg_train_loss = train_loss / len(train_loader)
        train_losses_epoch.append(avg_train_loss)
        
    return train_losses_epoch
        
