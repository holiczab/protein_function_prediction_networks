import torch, gc
import torch.nn as nn
import argparse
import numpy as np
import pickle
import time
from Utils import FC, Embedder, GCN
import Making_Graph
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from torchsummary import summary

gc.collect()
torch.cuda.empty_cache()

"""To save time and money, we included the results from SeqVec in the training dataset in advance."""

FOLDER = ""
SMALL_TYPE = ""


class CustomDataset(Dataset):
    def __init__(self, lable_map, train_data_file, valid_indexes=None):
        super(CustomDataset, self).__init__()
        self.idx_map = lable_map
        self.train_data_file = train_data_file
        self.valid_indexes = valid_indexes
        print(f"Training data file: {train_data_file}")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(self.train_data_file, 'rb') as f:
            data = pickle.load(f)
        seq = data["seqvec"].loc[idx]
        ant = data["annotations"].loc[idx][0]
        if self.valid_indexes is not None:
            annotations_list = data["annotations"].loc[idx][0].split(",")
            filtered_annotations = [annotation for annotation in annotations_list if annotation.strip() in self.valid_indexes]
            ant = ",".join(annotation.strip() for annotation in filtered_annotations)
        else:
            ant = data["annotations"].loc[idx][0]
        ant = ant.split(",")
        cls = [0] * len(self.idx_map)
        for a in ant:
            a = a.strip()
            cls[self.idx_map[a]] = 1
        cls = np.array(cls)
        cls = np.transpose(cls)
        return seq, cls

    def __len__(self):
        with open(self.train_data_file, 'rb') as f:
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

def train_model(args):
    print("training model...")
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    FOLDER = args.folder
    SMALL_TYPE = args.small_type

    # def preprocess_data(original_data_file, limited_data_file, max_size=25000):
    #     with open(original_data_file, 'rb') as f:
    #         data = pickle.load(f)

    #     # Shuffle data randomly
    #     shuffled_data = data.sample(frac=1).reset_index(drop=True)

    #     # Select limited data
    #     limited_data = shuffled_data.head(max_size)

    #     with open(limited_data_file, 'wb') as f:
    #         pickle.dump(limited_data, f)

    # original_data_file = args.type_path
    # limited_data_file = args.type_path.replace(f"train_data_{SMALL_TYPE}o.pkl",f"train_data_{SMALL_TYPE}o_25k.pkl")
    # preprocess_data(original_data_file, limited_data_file)

    # Data load
    adj, one_hot_node, label_map, label_map_ivs = Making_Graph.build_graph(FOLDER,SMALL_TYPE)

    def get_label_counts(dataset, valid_indexes=None):
      label_counts = {}
      total_samples = 0
      with open(dataset.train_data_file, 'rb') as f:
        data = pickle.load(f)
      total_samples = len(data)
      for idx in range(len(data)):
        if valid_indexes is not None:
            annotations_list = data["annotations"].loc[idx][0].split(",")
            filtered_annotations = [annotation for annotation in annotations_list if annotation.strip() in valid_indexes]
            ant = ",".join(annotation.strip() for annotation in filtered_annotations)
        else:
            ant = data["annotations"].loc[idx][0]
        for a in ant.strip().split(","):
          a = a.replace(" ", "")
          if a in label_counts:
            label_counts[a] += 1
          else:
            label_counts[a] = 1
      return label_counts, total_samples
    
    # Get label counts and total samples
    label_counts, total_samples = get_label_counts(CustomDataset(label_map, args.type_path))

    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    top_500_labels = [label for label, _ in sorted_labels[:500]]
    indexes = [idx for idx, label in enumerate(label_counts) if label in top_500_labels]

    # print(indexes)

    new_label_map = {key: value for key, value in label_map.items() if value in indexes}
    reindexed_label_map = {key: i for i, key in enumerate(new_label_map.keys())}

    # print(reindexed_label_map)
    #reindexed_label_map = label_map
    valid_indexes = new_label_map.keys()

    adj, one_hot_node, label_map, label_map_ivs = Making_Graph.build_graph(FOLDER,SMALL_TYPE, valid_indexes)

    data = CustomDataset(reindexed_label_map, args.type_path, valid_indexes)
    label_counts_new, total_samples = get_label_counts(data, valid_indexes)
    tr_dataset = data
    train_loader = DataLoader(dataset=tr_dataset, batch_size=args.batch_size)

    print("Len : Reindex_map")
    print(len(reindexed_label_map))
    print(len(label_counts_new))

    # Print total number of samples
    print(f"Total number of samples in training data: {total_samples}")
    
    # Create pie chart using matplotlib (assuming you have it installed)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.pie(label_counts_new.values(), labels=None, autopct="")
    plt.title("Label Distribution in Training Data")
    plt.savefig("C:/Users/T580/Desktop/ELTE/#SZAKDOLGOZAT/Effective-GCN-based-Hierarchical-Multi-label-classification-for-Protein-Function-Prediction/label_distribution.png")  # Replace with your desired filename

    print("Label distribution saved as label_distribution.png")
    
    # Save label counts to a text file
    with open("C:/Users/T580/Desktop/ELTE/#SZAKDOLGOZAT/Effective-GCN-based-Hierarchical-Multi-label-classification-for-Protein-Function-Prediction/label_counts.txt", "w") as f:
      for label, count in label_counts_new.items():
        f.write(f"{label}: {count}\n")

    print("Label counts saved to label_counts.txt")
    print(tr_dataset.__len__())

   # Load teacher model if available
    teacher_model = None  # Replace with loading your teacher model
    if args.teacher_model_path:
        teacher_model = Net(args).to(device)
        teacher_model.load_state_dict(torch.load(args.teacher_model_path))
        teacher_model.eval()  # Set to evaluation mode, no gradients

    # Initialize student model
    student_model = Net(len(reindexed_label_map), args).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Temperature for distillation
    T = 1.0  # Adjust this temperature as per your needs

    student_model.train()

    total_loss = 0
    print_every = 1
    start = time.time()
    temp = start
    
    for epoch in range(args.epochs):
        train_loss = 0

        for i, (input, target) in enumerate(train_loader):
            input = torch.stack(input).squeeze().to(device)
            target = target.type(torch.FloatTensor).to(device)
            one_hot_node = one_hot_node.to(device)
            adj = adj.to(device)

            optimizer.zero_grad()

            # Forward pass - student model
            student_logits = student_model(input, one_hot_node, adj)
            student_probs = torch.sigmoid(student_logits)

            # Calculate standard BCELoss
            loss = criterion(student_probs, target)

            # Knowledge Distillation - relation-based
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_relations = teacher_model.extract_relations(input)  # Modify according to your teacher model
                student_relations = student_model.extract_relations(input)  # Modify according to your student model

                # Example of relation-based loss (e.g., MSE)
                distillation_loss = F.mse_loss(student_relations, teacher_relations)
                loss += distillation_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_loss += loss.item()

            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print(f"time = {(time.time() - start) // 60}m, epoch {epoch + 1}, iter = {i + 1}, loss = {loss_avg:.3f}, {time.time() - temp:.3f}s per {print_every} iters")
                total_loss = 0
                temp = time.time()

        train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

        # Save model checkpoint after each epoch if needed

    # Save final student model
    torch.save(student_model.state_dict(), f'Weights/{FOLDER}/final_student.pth')


def main():
    parser = argparse.ArgumentParser()

    print("--- Training GCN ---")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Model parameters
    """BPO has 80 parameters, MFO has 41 parameters, and CCO has 54 parameters"""
    parser.add_argument("--nfeat", type=int, default=80, help="node feature size")
    parser.add_argument("--nhid", type=int, default=80, help="GCN node hidden size")
    parser.add_argument("--seqfeat", type=int, default=80, help="sequence reduced feature size")

    parser.add_argument("--type_path",type=str,default=f"C:/Users/T580/Desktop/ELTE/#SZAKDOLGOZAT/Effective-GCN-based-Hierarchical-Multi-label-classification-for-Protein-Function-Prediction/dataset/MFO/train_data_mfo_25k.pkl")
    #parser.add_argument("--type_path",type=str,default=f"C:/Users/T580/Desktop/ELTE/#SZAKDOLGOZAT/Effective-GCN-based-Hierarchical-Multi-label-classification-for-Protein-Function-Prediction/dataset/MFO/train_data_mfo.pkl")
    parser.add_argument("--folder",type=str,default="MFO")
    parser.add_argument("--small_type",type=str,default="mf")

    args = parser.parse_args()

    train_model(args)


if __name__ == '__main__':
    main()
