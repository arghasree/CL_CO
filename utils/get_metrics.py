import random
from typing import List, Dict
from cai2 import CAI, relative_adaptiveness
import torch
# import RNA
import csv
from utils.CONSTANTS import CODON_DICT as cds_token_dict
"""
Majority of the code is borrowed from : https://github.com/Adibvafa/CodonTransformer/blob/main/CodonTransformer/CodonEvaluation.py

"""

def get_CSI_weights(sequences: List[str]) -> Dict[str, float]:
    """
    Calculate the Codon Similarity Index (CSI) weights for a list of DNA sequences.

    Args:
        sequences (List[str]): List of DNA sequences.

    Returns:
        dict: The CSI weights.
    """
    return relative_adaptiveness(sequences=sequences)


def get_CSI_value(dna: str, weights: Dict[str, float]) -> float:
    """
    Calculate the Codon Similarity Index (CSI) for a DNA sequence.

    Args:
        dna (str): The DNA sequence.
        weights (dict): The CSI weights from get_CSI_weights.

    Returns:
        float: The CSI value.
    """
    return CAI(dna, weights)

def convert_index_to_codons(seq_pad_removed):
    codon_seq = ""
    #converting the codon to index dictionary to index to codon dictionary for inference of output sequences
    index_to_word = {v: k for k, v in cds_token_dict.items()}

    for index in seq_pad_removed:
        # index is of type tensor so need to convert to int
        if int(index) == -100:
            pass
            # codon_seq += index_to_word[random.choice([i for i in range(1, len(index_to_word))])]
        else:
            codon_seq += index_to_word[int(index)]
    return codon_seq

def get_batch_cai(output_seq_logits, cds_data_sorted, seq_lens, org_weights, test=False):
    """
    CAI: measures how well the output_seq_logits match the cds_data_sorted. 
    If it is high, it means the codon usage is similar to the GT. 
    """
    output_batch_cai = []
    target_batch_cai = []
    # print("Output Seq Logits Shape:", output_seq_logits.shape)
    predicted_output_logits = torch.argmax(output_seq_logits, dim=-1)
    # shape of output_seq_logits: (batch_size, seq_len, vocab_size)
    # shape: (batch_size, seq_len) 
    # [
    # [5, 12, 3, 42, 8],    # Batch 0: codon indices (5->ATG, 12->GGC, etc.)
    # [15, 8, 22, 1, 30]    # Batch 1: codon indices
    # ]

    for i in range(len(seq_lens)):
        # seq_lens is a list of lengths of each sequence in the batch
        # So there are 128 sequences in a batch
        # len(seq_lens) = 128
        # seq_lens[i] gives length of ith sequence in the batch
        trimmed_output_seq = predicted_output_logits[i][:seq_lens[i]]
       
        predicted_seq = convert_index_to_codons(trimmed_output_seq)
        output_batch_cai.append(CAI(predicted_seq, org_weights))
        # output_batch_cai.append(get_CSI_value(predicted_seq, org_weights))
        trimmed_target_seq = cds_data_sorted[i][:seq_lens[i]]
        target_seq = convert_index_to_codons(trimmed_target_seq)
        # target_batch_cai.append(get_CSI_value(target_seq, org_weights))
        target_batch_cai.append(CAI(target_seq, org_weights))
        
    # print("Output Batch CAI:", output_batch_cai[:3], "Target Batch CAI:", target_batch_cai[:3])
    return torch.tensor(output_batch_cai), torch.tensor(target_batch_cai)

