"""
Counts the number of Bethe-Heitler positrons
detected at the virtual detector

Timothy Chew
7/8/25
"""

from tqdm import tqdm

import math
import pandas as pd

input_file = "g4bl_stuff/g4beamlinesfiles/big_gamma_Det.txt"
output_file = "big_Gamma_Det_positrons.txt"

chunksize = 1000000  # adjust based on RAM
target_pdgid = -11

# Find column names first
with open(input_file) as f:
    f.readline()
    header = f.readline().strip().split()  # reads first line as col names

# Process in chunks
filtered_chunks = []
for chunk in tqdm(pd.read_csv(
    input_file,
    sep=r"\s+",
    names=header,
    comment='#',
    header=None,
    skiprows=1,         # skip header line already read
    chunksize=chunksize,
    dtype={col: "float32" for col in header if col != "PDGid"} | {"PDGid": "int32"},
    ),
    desc="Reading file",
    unit="chunk"
):
    filtered_chunk = chunk[chunk["PDGid"] == target_pdgid]
    filtered_chunks.append(filtered_chunk)

# Combine only the filtered chunks and write
pd.concat(filtered_chunks).to_csv(output_file, index=False)


# data = np.loadtxt('g4bl_stuff/g4beamlinesfiles/big_gamma_Det.txt', skiprows=1)
# mask = (data[:,7] == -11)
# print(len(data[mask]))
