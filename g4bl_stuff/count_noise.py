"""
Counts the number of Bethe-Heitler positrons
detected at the virtual detector

Timothy Chew
7/8/25
"""
import numpy as np
import pandas as pd

df = pd.read_csv('g4bl_stuff/g4bl_data/Gamma_profile_Det_deep_LWFA.txt', sep="\s+", header=1)
filtered_df = df[df['PDGid'] == -11]
print(filtered_df)
filtered_df.to_csv('TESTING.txt', index=False)
