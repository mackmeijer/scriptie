from dgl.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from PIL import Image
import random
import torch
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
import matplotlib
import matplotlib.pyplot
import dgl
import numpy as np
import torch
from torch.nn import Linear
import dgl.data
import matplotlib.pyplot as plt

def graph_gen():
    templates = [[0, 1], [2, 0], [2, 1], [2, 2]]
    seed = random.randint(0, 3)
    seed2 = random.randint(0, 3)

    images_list = [templates[seed][0], templates[seed][1], templates[seed2][0], templates[seed2][1]]
    x = [[x] for x in images_list]
    g = dgl.graph(([0, 1,  2,  3,  0,  1, 2,  3],
        [1, 0,  3,  2,  2,  3, 0,  1]), num_nodes=4)

    matrix_bo_on = [[None, 0, None], [None, None, None], [2, 2, 2]]
    matrix_on_bo = [[None, None, 4], [1, None, 4], [None, None, 3]]

    g.ndata['object'] = torch.zeros(4, 3)
    for i in range(0, 4):
        g.ndata['object'][i][x[i][0]] = 1

    g.edata['relation'] = torch.zeros(8, 7)
    g.edata['relation'][4][5] = 1
    g.edata['relation'][5][5] = 1
    g.edata['relation'][6][6] = 1
    g.edata['relation'][7][6] = 1


    g.edata['relation'][0][matrix_bo_on[x[0][0]][x[1][0]]] = 1
    g.edata['relation'][1][matrix_bo_on[x[2][0]][x[3][0]]] = 1
    g.edata['relation'][2][matrix_on_bo[x[1][0]][x[0][0]]] = 1
    g.edata['relation'][3][matrix_on_bo[x[3][0]][x[2][0]]] = 1

    if g.ndata['object'][0][0] == 1 or g.ndata['object'][1][0] == 1:
        label = torch.tensor([1])
    else:
        label = torch.tensor([0])

    return g, label


def get_batch(batch_size):
    x = []
    x_labels = []
    for i in range(batch_size):
        g, label = graph_gen()
        x.append(g)
        x_labels.append(label)

    x_labels = torch.concat(x_labels)
    return x, x_labels

class InformedSender(nn.Module):
    def __init__(self,  hidden_size, vocab_size, head_size):
        super(InformedSender, self).__init__()
        self.conv1 = GATConv(3, hidden_size, num_heads = head_size)
        self.conv2 = GATConv(hidden_size, 2, num_heads = head_size)
        self.lin1 = nn.Linear(hidden_size, 2)

    def forward(self, x, feat, _aux_input=None):
        res = self.conv1(x, feat)
        res= res.relu()
        res = self.conv2(x, res)
        x.ndata['h'] = res
        a = dgl.mean_nodes(x, 'h')
        print(a)
        print(a[0][0][0])
        raise(KeyboardInterrupt)
        return a[0][0][0]
    
dataset = dgl.data.GINDataset('PROTEINS', self_loop=True)


model = InformedSender(hidden_size= 4, vocab_size= 20, head_size= 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


acc_list = []


for epoch in range(100):
    batched_graph, labels = get_batch(5)
    for i in range(5):
        pred = model(batched_graph[i], batched_graph[i].ndata['object'])
        loss = F.cross_entropy(pred, labels[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    batched_graph, labels = get_batch(30)
    num_correct = 0
    num_tests = 0
    for i in range(30):
        pred = model(batched_graph[i], batched_graph[i].ndata['object'])
        if pred.argmax() == labels[i]:
            num_correct += 1
        num_tests += 1
    acc_list.append(num_correct / num_tests)

num_correct = 0
num_tests = 0
for epoch in range(10):
    batched_graph, labels = get_batch(5)
    for i in range(5):
        pred = model(batched_graph[i], batched_graph[i].ndata['object'])
        if pred.argmax() == labels[i]:
            num_correct += 1
        num_tests += 1

plt.plot(range(len(acc_list)), acc_list)
plt.show()
print('Test accuracy:', num_correct / num_tests)







