import json
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# from torch.nn.utils.rnn import pad_sequence
# from wandb import sklearn
# from data_preprocessing_util import DataPreprocessing
from utils.CONSTANTS import AMINO_ACID_DICT, CODON_DICT, ORGANISM_DICT
from utils.get_metrics import get_CSI_weights

random.seed(42)

import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



class AminoAcidCodonDataset(Dataset):
    def __init__(self, X, codon_sequences, max_len):
        self.aa_sequences = [x[0] for x in X][:200]
        self.organism_list = [x[1] for x in X][:200]
        self.codon_sequences = codon_sequences[:200]
        self.max_len = max_len

    def __len__(self):
        return len(self.aa_sequences)

    def tokenize_aa_sequence(self, aa_seq):
        return [AMINO_ACID_DICT[aa] for aa in aa_seq]  # Map amino acids to integers

    def tokenize_codon_sequence(self, codon_seq):
        codons = [codon_seq[i:i+3] for i in range(0, len(codon_seq), 3)]
        # print(codons, " --- codons --- ")
        return [CODON_DICT.get(codon.lower(), 0) for codon in codons]  # Map codons to integers
    
    def tokenize_organism(self, organism):
        return ORGANISM_DICT.get(organism, 0)  # Map organism to integers
        

    def create_attention_mask(self, seq_length):
        return [1] * seq_length + [0] * (self.max_len - seq_length)

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self.aa_sequences)), sample_size)
        # take sample 
        # sampled_aa = [self.aa_sequences[i] for i in sample_idx]
        # sampled_codon = [self.codon_sequences[i] for i in sample_idx]
        # organism = [self.organism_list[i] for i in sample_idx]
        sampled_aa = [self.aa_sequences[i] for i in sample_idx]
        sampled_codon = [self.codon_sequences[i] for i in sample_idx]
        sampled_organism = [self.organism_list[i] for i in sample_idx]
        # print("Sampled organisms:", sampled_organism)
        
        max_len_a = max(len(seq) for seq in sampled_aa)
        max_len_c = max(len(seq)//3 for seq in sampled_codon)
        
        sampled_aa = [self.tokenize_aa_sequence(aa_seq) for aa_seq in sampled_aa]
        sampled_codon = [self.tokenize_codon_sequence(codon_seq) for codon_seq in sampled_codon]
        sampled_organism = [self.tokenize_organism(org) for org in sampled_organism]
        # print("Sampled organism tokens:", sampled_organism)
        sampled_aa = [aa_tokens + [0] * (max_len_a - len(aa_tokens)) for aa_tokens in sampled_aa]
        sampled_codon = [codon_tokens + [-100] * (max_len_c - len(codon_tokens)) for codon_tokens in sampled_codon]

        return torch.tensor(sampled_aa, dtype=torch.long), torch.tensor(sampled_codon, dtype=torch.long), torch.tensor(sampled_organism, dtype=torch.long)
        # return sampled_aa[:self.max_len], sampled_codon[:self.max_len], sampled_organism [:self.max_len]       
        # # padding
        # sampled_aa_padded = []
        # sampled_codon_padded = []
        # for aa_tokens, codon_tokens in zip(sampled_aa, sampled_codon):
        #     aa_tokens += [0] * (self.max_len - len(aa_tokens))
        #     codon_tokens += [-100] * (self.max_len - len(codon_tokens))
        #     sampled_aa_padded.append(aa_tokens[:self.max_len])
        #     sampled_codon_padded.append(codon_tokens[:self.max_len])
        
        # return {
        #     'input_ids': torch.tensor(sampled_aa_padded, dtype=torch.long),
        #     'organism_id': torch.tensor(sampled_organism, dtype=torch.long),
        #     'labels': torch.tensor(sampled_codon_padded, dtype=torch.long)
        # }


    def __getitem__(self, idx):
        aa_seq = self.aa_sequences[idx]
        codon_seq = self.codon_sequences[idx]
        organism = self.organism_list[idx]
        aa_tokens = self.tokenize_aa_sequence(aa_seq)
        codon_tokens = self.tokenize_codon_sequence(codon_seq)
        organism_token = self.tokenize_organism(organism)
        # Create attention mask
        attention_mask = self.create_attention_mask(len(aa_tokens))

        # Padding
        aa_tokens += [0] * (self.max_len - len(aa_tokens))
        codon_tokens += [-100] * (self.max_len - len(codon_tokens))

        return {
            'input_ids': torch.tensor(aa_tokens[:self.max_len], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'organism_id': torch.tensor(organism_token, dtype=torch.long),
            'labels': torch.tensor(codon_tokens[:self.max_len], dtype=torch.long)
        }



def start_preprocessing(data_file_path):
    # read data from file
    df = pd.read_csv(data_file_path)
    ref_cds_list = list(df['dna'])
    # Only select the rows with normalized_mfe <=-0.3
    # df = df[df['normalized_mfe'] < -0.2]
    
    cds_list = list(df['dna'])
    aa_list = list(df['protein'])
    if 'hg19' in data_file_path:
        organism_list = ['hg19'] * len(aa_list)
    else:
        organism_list = list(df['organism'])
    

    cds_list = [x[:-3] for x in cds_list]  # remove stop codon
    aa_list = [x[:-1] for x in aa_list]  # remove unknown amino acid corresponding to stop codon
    
    max_len = max(len(cds_seq) for cds_seq in cds_list) # max length in terms of codons
    # X = np.stack([aa_list, organism_list], axis=0)
    # Y = cds_list
    X = list(zip(aa_list, organism_list))
    Y = cds_list
    
    # train_val_aa, test_aa, train_val_cds, test_cds = train_test_split(aa_list, cds_list, test_size=0.2)

    # train_aa, val_aa, train_cds, val_cds = train_test_split(train_val_aa, train_val_cds, test_size=0.2)
    train_val_aa, test_aa, train_val_cds, test_cds = train_test_split(X, Y, test_size=0.2)
    train_aa, val_aa, train_cds, val_cds = train_test_split(train_val_aa, train_val_cds, test_size=0.2)

    train_dataset = AminoAcidCodonDataset(train_aa, train_cds, max_len)
    val_dataset = AminoAcidCodonDataset(val_aa, val_cds, max_len)
    test_dataset = AminoAcidCodonDataset(test_aa, test_cds, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    print(f'Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}')

    # get weights value for this organism
    org_weights = get_CSI_weights(sequences=ref_cds_list)
    # print("Calculated organism weights for loss function.", org_weights)
    return train_loader, val_loader, test_loader, org_weights

if __name__ == '__main__':
    datafile_path = './cl_dataset/organism=Homo sapiens.csv'
    train_loader, val_loader, test_loader = start_preprocessing(datafile_path)
    print(f'Length of train_loader: {len(train_loader.dataset)}')
    print(f'Length of val_loader: {len(val_loader.dataset)}')
    print(f'Length of test_loader: {len(test_loader.dataset)}')