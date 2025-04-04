import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
from dgl.data import DGLDataset



# Model definition
def MLP(channels):
    return nn.Sequential(*[
        nn.Sequential(nn.BatchNorm1d(channels[i - 1]),
                      nn.Linear(channels[i - 1], channels[i]),
                      nn.ReLU())
        for i in range(1, len(channels))
    ])


class EdgeConv(nn.Module):
    def __init__(self, mlp1, mlp2):
        super(EdgeConv, self).__init__()
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        # self.reset_parameters()

    def concat_message_function(self, edges):

        b = edges.data['feature']
        c = edges.src['hid']

        cat = torch.cat((b, c), axis=1)
        # print(cat.shape)
        return {'out': self.mlp1(cat)}

    def apply_func(self, nodes):
        a = nodes.data['reduced_vector']
        b = nodes.data['hid']

        cat = torch.cat((a, b), axis=1)

        return {"hid": self.mlp2(cat)}

    def forward(self, g):
        g.apply_edges(self.concat_message_function)

        g.update_all(fn.copy_e('out', 'msg'),
                     fn.mean('msg', 'reduced_vector'),
                     self.apply_func)


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        self.conv1 = EdgeConv(MLP([3, 9]), MLP([10, 10]))
        self.conv2 = EdgeConv(MLP([12, 10]), MLP([11 + 9, 10]))
        self.conv3 = EdgeConv(MLP([12, 10]), MLP([11 + 9, 10]))
        self.conv4 = EdgeConv(MLP([12, 10]), MLP([11 + 9, 10]))



    def forward(self, g):
        g.ndata['hid'] = g.ndata['feature']  # initialization of GNN
        self.conv1(g)
        self.conv2(g)
        self.conv3(g)
        self.conv4(g)

        return g.ndata['hid']


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.gcn1 = GCN()
        self.gcn2 = GCN()
        self.gcn3 = GCN()
        self.gcn4 = GCN()
        self.gcn5 = GCN()
        self.gcn6 = GCN()
        self.gcn7 = GCN()
        self.gcn8 = GCN()
        self.gcn9 = GCN()
        self.gcn10 = GCN()

        self.embedding_layers = [self.gcn1,
                                 self.gcn2,
                                 self.gcn3,
                                 self.gcn4,
                                 self.gcn5,
                                 self.gcn6,
                                 self.gcn7,
                                 self.gcn8,
                                 self.gcn9,
                                 self.gcn10]


        self.cnn = nn.Sequential(
            nn.Conv2d(10, 5, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(5, 2, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(2, 1, kernel_size=(3, 3), stride=1, padding=1)
        )
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g_list):
        batch_size = g_list[0].batch_size
        num_resource_blocks = len(g_list)
        
        # Apply GCN to each batched graph in parallel
        X = [self.embedding_layers[k](g_list[k]) for k in range(num_resource_blocks)]
        # Stack and permute for CNN input
        X_stacked = torch.stack(X, dim=0)
        CNN_input = X_stacked.permute(2, 1, 0)
        
        # Apply CNN and reshape to [batch_size, num_resource_blocks, features]
        A = self.cnn(CNN_input).reshape(batch_size, num_resource_blocks, -1)
        
        # Apply sigmoid and calculate p_train
        A_sigmoid = self.sigmoid(A)
        p_train = torch.sum(A_sigmoid, dim=1) / num_resource_blocks
        
        # Apply softmax along the resource blocks dimension (dim=1)
        # This ensures softmax is applied independently for each sample
        rb_train = F.softmax(A, dim=1)
        
        return p_train, rb_train



if __name__ == '__main__':

    
    # Working device (either cpu or gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Working on:", device)

    # Define the model
    model = Model().to(device)
    print(model)