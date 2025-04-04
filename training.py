import os
import random
from datetime import datetime
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
from dgl.data import DGLDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets import Model
from torch.utils.data import random_split
from tqdm import tqdm
from utils import build_graph


class PCDataset(DGLDataset):

    def process(self):
        pass

    def __init__(self, data_root):
        # extract paths
        self.csi_path = os.path.join(data_root, "csi")
        self.p_path = os.path.join(data_root, "p")
        self.rb_path = os.path.join(data_root, "rb")
        self.samples_list_path = os.path.join(data_root, "samples_list.csv")

        # get samples_list
        self.samples_list = np.genfromtxt(self.samples_list_path, delimiter=',')


        super().__init__(name='power_bandwidth_control')


    def __len__(self):
        """Denotes the total number of samples"""
        # print(int(0.1 * len(self.samples_list)) - 1)
        return int(1 * len(self.samples_list)) - 1

    def __getitem__(self, index):
        """Generates one sample of data"""
        # CSI
        csi = torch.from_numpy(np.load(os.path.join(self.csi_path, "csi_" + str(index) + ".npy"))).float()
        # Power and Bandwidth allocation
        p = torch.from_numpy(np.load(os.path.join(self.p_path, "p_" + str(index) + ".npy"))).float()
        rb = torch.from_numpy(np.load(os.path.join(self.rb_path, "rb_" + str(index) + ".npy"))).long()

        graph_list = build_graph(csi)

        return graph_list, csi, rb, p


def collate(samples):
    """DGL collate function"""
    K = len(samples[0][0]) # number of resource blocks
    # print("K : ", K)
    batched_p = []
    batched_rb = []
    batched_csi = []
    batched_g_list = [[] for _ in range(K)]

    for g_list, csi, rb, p in samples:
        batched_csi.append(csi)
        batched_rb.append(rb)
        batched_p.append(p)

        for i in range(len(g_list)):
            batched_g_list[i].append(g_list[i])


    # print("batched_g_list : ", len(batched_g_list))
    # for i in range(len(batched_g_list)):
    #     print("batched_g_list[{}] : ".format(i), len(batched_g_list[i]))

    # Stack the tensors
    batched_csi = torch.stack(batched_csi, dim=0)
    batched_rb = torch.stack(batched_rb, dim=0)
    batched_p = torch.stack(batched_p, dim=0)

    # Create a batched graph
    batched_graphs = [dgl.batch(graphs) for graphs in batched_g_list]
    
    return batched_graphs, batched_csi, batched_rb, batched_p


if __name__ == '__main__':
    # Define network parameters
    net_par = {
        "d0": 1,
        'htx': 1.5,
        'hrx': 1.5,
        'antenna_gain_decibel': 2.5,
        'noise_density_milli_decibel': -169,
        'carrier_f': 2.4e9,
        'shadow_std': 8,
        "rb_bandwidth": 5e2,
        "wc": 50,
        "wd": 20,
        "wx": 500,
        "wy": 500,
        "N": 50,  # Number of links
        "K": 10    # Number of resource blocks
    }

    # Working device (either cpu or gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Working on:", device)

    # Define the model
    model = Model().to(device)
    # print(model)

    # Configure datasets and dataloaders
    train_data_root = "DATASET_train_10000_p_1_c=1e3_N=50_rb=10"
    full_data = PCDataset(train_data_root)

    # Split the dataset into training and validation sets
    train_size = int(0.95 * len(full_data))
    test_size = len(full_data) - train_size


    # Split the dataset
    train_data, test_data = random_split(full_data, [train_size, test_size])
    print(f"Training samples: {len(train_data)}, Validation samples: {len(test_data)}")

    # Create DataLoader for training and validation
    train_data = PCDataset(train_data_root)
    test_data = PCDataset(train_data_root)
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True, collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False, collate_fn=collate, pin_memory=True)

    # Print the number of batches
    print(f"Number of batches in training: {len(train_loader)}")
    print(f"Number of batches in validation: {len(test_loader)}")

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Define loss functions
    mse_loss = nn.MSELoss()
    
    # Training parameters
    num_epochs = 300
    print_every = 10  # Print loss every 10 batches
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0.0
        p_loss_total = 0.0
        rb_loss_total = 0.0
        
        for i, sample in enumerate(train_loader):
            batched_graphs, batched_csi, batched_rb, batched_p = sample
            batched_graphs = [g.to(device) for g in batched_graphs]
            batched_csi = batched_csi.to(device)
            batched_rb = batched_rb.to(device)
            batched_p = batched_p.to(device)

            optimizer.zero_grad()
            p_pred, rb_pred = model(batched_graphs)

            p_loss = mse_loss(p_pred, batched_p)
            log_probs = torch.log(rb_pred + 1e-8)
            log_probs = log_probs.permute(0, 2, 1).reshape(-1, 10)
            targets = batched_rb.reshape(-1)
            rb_loss = F.nll_loss(log_probs, targets)

            loss = p_loss + 0.1 * rb_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            p_loss_total += p_loss.item()
            rb_loss_total += rb_loss.item()

            if (i + 1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, P Loss: {p_loss.item():.4f}, RB Loss: {rb_loss.item():.4f}")
        
        # Training epoch stats
        avg_loss = epoch_loss / len(train_loader)
        avg_p_loss = p_loss_total / len(train_loader)
        avg_rb_loss = rb_loss_total / len(train_loader)

        # --- Testing Phase ---
        model.eval()
        test_loss, test_p_loss, test_rb_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for sample in test_loader:
                batched_graphs, batched_csi, batched_rb, batched_p = sample
                batched_graphs = [g.to(device) for g in batched_graphs]
                batched_csi = batched_csi.to(device)
                batched_rb = batched_rb.to(device)
                batched_p = batched_p.to(device)

                p_pred, rb_pred = model(batched_graphs)

                p_loss = mse_loss(p_pred, batched_p)
                log_probs = torch.log(rb_pred + 1e-8)
                log_probs = log_probs.permute(0, 2, 1).reshape(-1, 10)
                targets = batched_rb.reshape(-1)
                rb_loss = F.nll_loss(log_probs, targets)

                loss = p_loss + 0.1 * rb_loss
                test_loss += loss.item()
                test_p_loss += p_loss.item()
                test_rb_loss += rb_loss.item()

        avg_test_loss = test_loss / len(test_loader)
        avg_test_p_loss = test_p_loss / len(test_loader)
        avg_test_rb_loss = test_rb_loss / len(test_loader)

        # Print combined epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train — Loss: {avg_loss:.4f}, P Loss: {avg_p_loss:.4f}, RB Loss: {avg_rb_loss:.4f}")
        print(f"  Test  — Loss: {avg_test_loss:.4f}, P Loss: {avg_test_p_loss:.4f}, RB Loss: {avg_test_rb_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"model_checkpoint_epoch_{epoch+1}.pt")



    # Save final model
    torch.save(model.state_dict(), "final_model.pt")
    print("Training completed!")