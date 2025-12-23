# Install ViennaRNA if not already installed
# !pip install ViennaRNA

import RNA
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm as tqdm
from pathlib import Path
from utils.CONSTANTS import ORGANISMS
import os

def get_mfe_stability(sequence):
    fc  = RNA.fold_compound(sequence)
    (_, mfe) = fc.mfe()
    return mfe


base_dir = "./cl_dataset"
# datsets_path = []

for organism in ORGANISMS:
    data_path =os.path.join(base_dir, f"organism={organism}.csv")
    df = pd.read_csv(data_path)
    df['mfe'] = df['dna'].apply(get_mfe_stability)
    df['normalized_mfe'] = df['mfe'] / df['dna'].str.len()
    df.to_csv(data_path, index=False)
    print(f'Stability calculate and saved for the {organism}')
