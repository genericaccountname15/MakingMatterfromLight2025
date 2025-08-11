import uproot

file = uproot.open("g4beamline.root")
tree = file["VirtualDetector/noise_measure_Det;1"]  # usually the tree inside
df = tree.arrays(library="pd")
print(df.head())