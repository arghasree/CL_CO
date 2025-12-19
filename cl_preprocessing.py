import json
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# from torch.nn.utils.rnn import pad_sequence
# from wandb import sklearn
# from data_preprocessing_util import DataPreprocessing
from utils.CONSTANTS import AMINO_ACID_DICT, CODON_DICT, SYNONYMOUS_CODONS
from utils.get_metrics import get_CSI_weights

random.seed(42)

import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



class AminoAcidCodonDataset(Dataset):
    def __init__(self, X, codon_sequences, max_len):
        self.aa_sequences = [x[0] for x in X][:500]
        self.organism_list = [x[1] for x in X][:500]
        self.codon_sequences = codon_sequences[:500]
        self.max_len = max_len

    def __len__(self):
        return len(self.aa_sequences)

    def tokenize_aa_sequence(self, aa_seq):
        return [AMINO_ACID_DICT[aa] for aa in aa_seq]  # Map amino acids to integers

    def tokenize_codon_sequence(self, codon_seq):
        codons = [codon_seq[i:i+3] for i in range(0, len(codon_seq), 3)]
        return [CODON_DICT.get(codon, 0) for codon in codons]  # Map codons to integers
    def tokenize_organism(self, organism):
        pass #TODO: implement organism tokenization if needed

    def create_attention_mask(self, seq_length):
        return [1] * seq_length + [0] * (self.max_len - seq_length)

    def __getitem__(self, idx):
        aa_seq = self.aa_sequences[idx]
        codon_seq = self.codon_sequences[idx]

        aa_tokens = self.tokenize_aa_sequence(aa_seq)
        codon_tokens = self.tokenize_codon_sequence(codon_seq)

        # Create attention mask
        attention_mask = self.create_attention_mask(len(aa_tokens))

        # Padding
        aa_tokens += [0] * (self.max_len - len(aa_tokens))
        codon_tokens += [-100] * (self.max_len - len(codon_tokens))

        return {
            'input_ids': torch.tensor(aa_tokens[:self.max_len], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(codon_tokens[:self.max_len], dtype=torch.long)
        }



def start_preprocessing(data_file_path):
    # read data from file
    df = pd.read_csv(data_file_path)
    cds_list = list(df['dna'])
    aa_list = list(df['protein'])
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
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # get weights value for this organism
    org_weights = get_CSI_weights(sequences=cds_list)
    return train_loader, val_loader, test_loader, org_weights

if __name__ == '__main__':
    datafile_path = './cl_dataset/organism=Homo sapiens.csv'
    train_loader, val_loader, test_loader = start_preprocessing(datafile_path)
    print(f'Length of train_loader: {len(train_loader.dataset)}')
    print(f'Length of val_loader: {len(val_loader.dataset)}')
    print(f'Length of test_loader: {len(test_loader.dataset)}')