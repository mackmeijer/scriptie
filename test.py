from dgl.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F
import sys
from PIL import Image
import random
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
import egg.core as core
import argparse
import cv2





class InformedSender(nn.Module):
    def __init__(
        self, game_size, feat_size, embedding_size, hidden_size, vocab_size=10, temp=1.0, head_size = 2):
        super(InformedSender, self).__init__()
        self.game_size = 2
        self.embedding_size = 30
        self.hidden_size = 80
        self.vocab_size = vocab_size
        self.temp = temp

        self.im_lin1 = nn.Linear(feat_size, embedding_size, bias=False)
        self.im_conv2 = nn.Conv2d( 1, hidden_size,  kernel_size=(game_size, 1), stride=(game_size, 1), bias=False,)
        self.im_conv3 = nn.Conv2d(1, 1, kernel_size=(hidden_size, 1), stride=(hidden_size, 1), bias=False)
        self.im_lin4 = nn.Linear(embedding_size * 2, vocab_size, bias=False)
        self.gr_conv1 = GATConv(3, hidden_size, num_heads = head_size)
        self.gr_conv2 = GATConv(hidden_size, embedding_size, num_heads = head_size)
        self.gr_lin1 = nn.Linear(hidden_size, 2)

    def forward(self, x, g, feat, _aux_input=None):
        emb = self.return_embeddings(x)

        h = self.im_conv2(emb)
        h = torch.sigmoid(h)
        # in: h of size (batch_size, hidden_size, 1, embedding_size)
        # out: h of size (batch_size, 1, hidden_size, embedding_size)
        h = h.transpose(1, 2)
        h = self.im_conv3(h)
        # h of size (batch_size, 1, 1, embedding_size)
        h = torch.sigmoid(h)
        h = h.squeeze(dim=1)
        h = h.squeeze(dim=1)
        # h of size (batch_size, embedding_size)
        res = self.gr_conv1(g, feat)
        res= res.relu()
        res = self.gr_conv2(g, res)
        g.ndata['h'] = res
        a = dgl.mean_nodes(g, 'h')
        a = dgl.mean_nodes(g, 'h')
        h = torch.cat((a[0][0][0], h[0]))
        h = self.im_lin4(h)
        h = h.mul(1.0 / self.temp)
        # h of size (batch_size, vocab_size)
        logits = F.log_softmax(h, dim=0)
        return logits

    def return_embeddings(self, x):
        # Embed each image (left or right)
        embs = []
        h = torch.from_numpy(x).float()
        
        # Assuming h has a size of [400 x 400], you may want to flatten it
        h = h.view(-1)
        # Apply the linear layer
        h_i = self.im_lin1(h)
        
        # Reshape the result to [batch_size x 1 x 1 x embedding_size]
        h_i = h_i.view(-1, 1, 1, self.embedding_size)
        
        # Append to the list of embeddings
        embs.append(h_i)
        
        # Concatenate the embeddings along the fourth dimension
        h = torch.cat(embs, dim=2)
        print(h.size())
        return h

class Receiver(nn.Module):
    def __init__(
        self, game_size, feat_size, embedding_size, hidden_size, vocab_size=10, temp=1.0, head_size = 2):
        super(Receiver, self).__init__()
        self.game_size = 2
        self.embedding_size = 30
        self.hidden_size = 80
        self.vocab_size = vocab_size
        self.temp = temp
        
        self.im_lin1 = nn.Linear(feat_size, embedding_size, bias=False)
        self.im_conv2 = nn.Conv2d( 1, hidden_size,  kernel_size=(game_size, 1), stride=(game_size, 1), bias=False,)
        self.im_conv3 = nn.Conv2d(1, 1, kernel_size=(hidden_size, 1), stride=(hidden_size, 1), bias=False)
        self.im_lin4 = nn.Linear(embedding_size * 2 + vocab_size, 2, bias=False)
        self.gr_conv1 = GATConv(3, hidden_size, num_heads = head_size)
        self.gr_conv2 = GATConv(hidden_size, embedding_size, num_heads = head_size)
        self.gr_lin1 = nn.Linear(hidden_size, 2)

    def forward(self, x, g, feat, vocab, _aux_input=None):
        emb = self.return_embeddings(x)
        h = self.im_conv2(emb)
        h = torch.sigmoid(h)
        # in: h of size (batch_size, hidden_size, 1, embedding_size)
        # out: h of size (batch_size, 1, hidden_size, embedding_size)
        h = h.transpose(1, 2)
        h = self.im_conv3(h)
        # h of size (batch_size, 1, 1, embedding_size)
        h = torch.sigmoid(h)
        h = h.squeeze(dim=1)
        h = h.squeeze(dim=1)
        # h of size (batch_size, embedding_size)

        res = self.gr_conv1(g, feat)
        res= res.relu()
        res = self.gr_conv2(g, res)
        g.ndata['h'] = res
        a = dgl.mean_nodes(g, 'h')
        a = dgl.mean_nodes(g, 'h')
        h = torch.cat((a[0][0][0], h[0]))
        h = torch.cat((h, vocab))
        h = self.im_lin4(h)
        
        h = h.mul(1.0 / self.temp)
        # h of size (batch_size, vocab_size)
        logits = F.log_softmax(h, dim=0)
        print(logits)
        return logits

    def return_embeddings(self, x):
        # Embed each image (left or right)
        embs = []
        h = torch.from_numpy(x).float()
        
        # Assuming h has a size of [400 x 400], you may want to flatten it
        h = h.view(-1)
        # Apply the linear layer
        h_i = self.im_lin1(h)
        
        # Reshape the result to [batch_size x 1 x 1 x embedding_size]
        h_i = h_i.view(-1, 1, 1, self.embedding_size)
        
        # Append to the list of embeddings
        embs.append(h_i)
        
        # Concatenate the embeddings along the fourth dimension
        h = torch.cat(embs, dim=2)
        print(h.size())
        return h

def pic_gen():
    templates = [[0, 1], [2, 0], [2, 1], [2, 2]]
    images = [[r".\img\fles.png", r".\img\lama.png", r".\img\tulp.png"], [r".\img\pilaar.png", r".\img\tafel.png", r".\img\stoel.png"], [r".\img\leeg.png", r".\img\leeg.png", r".\img\leeg.png"]]

    seed = random.randint(0, 3)
    seed2 = random.randint(0, 3)

    images_list = [templates[seed][0], templates[seed][1], templates[seed2][0], templates[seed2][1]]
    x = [[x] for x in images_list]
    images_list = [images[images_list[0]][random.randint(0, 2)], images[images_list[1]][random.randint(0, 2)], images[images_list[2]][random.randint(0, 2)], images[images_list[3]][random.randint(0, 2)]]
    images = [Image.open(x) for x in images_list]

    widths, heights = images[0].size

    total_width = widths*2
    max_height = heights*2

    new_im = Image.new('RGB', (total_width, max_height))

    xy = [[0, 0], [0, 1], [1, 0], [1, 1]]
    offset = 0
    for im in images:
        x_offset, y_offset = im.size[0]*xy[offset][0], im.size[0]*xy[offset][1]
        new_im.paste(im, (x_offset, y_offset))
        offset+=1
    new_im.save('test.jpg')
    new_im = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)


    if images_list[0] == '.\\img\\leeg.png' and images_list[1] == '.\\img\\leeg.png':
        label = torch.tensor([1])
    else:
        label = torch.tensor([0])

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

    return new_im, g

def get_batch(batch_size):
    images = []
    graphs = []
    for i in range(batch_size):
        img, g = pic_gen()
        images.append(img)
        graphs.append(g)

    return images, graphs

model = InformedSender(game_size = 1, feat_size = 160000, embedding_size = 30, hidden_size=80, head_size = 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
images, graphs = pic_gen()
pred = model(images, graphs, graphs.ndata['object'])

model2 = Receiver(game_size = 1, feat_size = 160000, embedding_size = 30, hidden_size=80, head_size = 2)
pred2 = model2(images, graphs, graphs.ndata['object'], pred)
print("pred", pred2)