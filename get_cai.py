import pandas as pd
from collections import Counter
from tqdm.auto import tqdm as tqdm
from pathlib import Path
from utils.CONSTANTS import ORGANISMS
import os
from cai2 import CAI, relative_adaptiveness

def get_CAI(sequence, weights):
    return CAI(sequence, weights)



base_dir = "./cl_dataset"
ORGANISMS = ORGANISMS[:3]
for organism in ORGANISMS:
    data_path =os.path.join(base_dir, f"organism={organism}.csv")
    df = pd.read_csv(data_path)
    codon_sequences = df['dna'].tolist()
    cai_weights = relative_adaptiveness(codon_sequences)
    df['cai'] = df['dna'].apply(lambda seq: get_CAI(seq, cai_weights))
    df.to_csv(data_path, index=False)
    print(f'CAI calculated and saved for the {organism}')

