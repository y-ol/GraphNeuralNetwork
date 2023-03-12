import pickle

with open("./dataset/qm9/graphs.pickle", "wb") as f:
    pickle.dump(otherDataset.dataset.graphs, f)
with open("./dataset/qm9/graphs.pickle", "rb") as f:
    graphs = pickle.load(f)
with open("")
