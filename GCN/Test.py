import torch, gc
import torch.nn as nn
import torch.optim as optim
import Making_Graph
from Utils import FC, Embedder, GCN
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset, Subset
import argparse

gc.collect()
torch.cuda.empty_cache()

FOLDER = ""
SMALL_TYPE = ""

class CustomDataset(Dataset):
    def __init__(self, test_data_file):
        super(CustomDataset, self).__init__()
        self.test_data_file = test_data_file
        print(f"Testing data file: {test_data_file}")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with open(self.test_data_file, 'rb') as f:
            data = pickle.load(f)
        seq = data["seqvec"].loc[idx]
        seq = np.array(seq)
        seq = np.transpose(seq)
        id = data["proteins"].loc[idx]
        return seq, id

    def __len__(self):
        with open(self.test_data_file, 'rb') as f:
            data = pickle.load(f)
        return len(data)


class Net(nn.Module):
    def __init__(self, n_class, args):
        super().__init__()
        self.FC = FC(1024, args.seqfeat)
        self.Graph_Embedder = Embedder(n_class, args.nfeat)
        self.GCN = GCN(args.nfeat, args.nhid)

    def forward(self, seq, node, adj):
        seq_out = self.FC(seq)
        node_embd = self.Graph_Embedder(node)
        graph_out = self.GCN(node_embd, adj)
        graph_out = graph_out.transpose(-2, -1)
        output = torch.matmul(seq_out, graph_out)
        output = torch.sigmoid(output)
        return output


def test_model(args):

    FOLDER = args.folder
    SMALL_TYPE = args.small_type

    adj, one_hot_node, label_map, label_map_ivs = Making_Graph.build_graph(FOLDER,SMALL_TYPE)

    test_dataset = CustomDataset(args.type_path)

    test_dataset = CustomDataset(args.type_path)
    subset_indices = list(range(min(len(test_dataset), 1000)))
    test_subset = Subset(test_dataset, subset_indices)

    test_loader = DataLoader(dataset=test_subset, batch_size=args.batch_size)

    device = torch.device("cpu")
    print(device)

    model = torch.load("Weights/bpo_final.pth", map_location=torch.device('cpu'))
    model = model.eval()

    yy = []
    for i, (input, id) in enumerate(test_loader):
        input = input.squeeze().to(device)
        input = input.type(torch.float32)
        one_hot_node = one_hot_node.to(device)
        adj = adj.to(device)
        preds = model(input, one_hot_node, adj)
        preds = preds.tolist()
        if len(preds) == 1 or i == 67:
            yy.append(preds)
        else:
            for j in range(len(preds)):
                yy.append(preds[j])

    np.save("bpo_preds.npy", yy)
    print("finish")


def main():
    parser = argparse.ArgumentParser()

    print("--- Testing GCN ---")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--dropout', type=int, default=0.1)

    # Model parameters
    """BPO has 80 parameters, MFO has 41 parameters, and CCO has 54 parameters"""
    parser.add_argument("--nfeat", type=int, default=80, help="node feature size")
    parser.add_argument("--nhid", type=int, default=80, help="GCN node hidden size")
    parser.add_argument("--seqfeat", type=int, default=80, help="sequence reduced feature size")

    parser.add_argument("--type_path",type=str,default="dataset/MFO/test_data_mfo.pkl")
    parser.add_argument("--folder",type=str,default="MFO")
    parser.add_argument("--small_type",type=str,default="mf")

    args = parser.parse_args()

    test_model(args)


if __name__ == '__main__':
    main()
