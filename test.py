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
        self.game_size = game_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.temp = temp

        self.im_lin1 = nn.Linear(feat_size, embedding_size, bias=False)
        self.im_conv1 = nn.Conv2d(1, hidden_size, kernel_size=(game_size, 1), stride=(game_size, 1), bias=False)
        self.im_conv2 = nn.Conv2d(1, 1, kernel_size=(hidden_size, 1), stride=(hidden_size, 1), bias=False)
        self.im_lin2 = nn.Linear(embedding_size * 2, vocab_size, bias=False)

        self.gr_conv1 = GATConv(3, hidden_size, num_heads = head_size)
        self.gr_conv2 = GATConv(hidden_size, embedding_size, num_heads = head_size)
        self.gr_lin1 = nn.Linear(embedding_size*2, embedding_size)

    def forward(self, x, graphs, _aux_input=None):
        #embedding images 
        emb = []
        for i in range(self.game_size):
            emb.append(self.return_embeddings(x[i]))
        emb = torch.concat(emb, dim=2)       # batch_size, 1, game_size, embedding_size


        h = self.im_conv1(emb)               # batch_size, hidden_size, 1, embedding_size
        h = torch.nn.LeakyReLU()(h)

        h = h.transpose(1, 2)                # batch_size, 1, hidden_size, embedding_size
        h = self.im_conv2(h)                 # 1, 1, 1, embedding_size
        h = torch.nn.LeakyReLU()(h)

        h = h.squeeze(dim=1)
        h = h.squeeze(dim=1)                 # h of size (batch_size, embedding_size)

        
        #embedding graphs
        gr_emb = []
        for i in range(self.game_size):
            g = graphs[i]
            feat = g.ndata['object']

            res = self.gr_conv1(g, feat)
            res= res.relu()
            res = self.gr_conv2(g, res)
            g.ndata['h'] = res
            a = dgl.mean_nodes(g, 'h')
            a = dgl.mean_nodes(g, 'h')
            gr_emb.append(a[0][0][0])

        gr_emb = torch.concat(gr_emb, dim=0)  #emb_size * 2

        gr_emb = self.gr_lin1(gr_emb)         #emb_size

        h = torch.cat((gr_emb, h[0]))         #emb_size * 2

        h = self.im_lin2(h)                   #vocab_size
        h = h.mul(1.0 / self.temp)
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
        self.im_conv1 = nn.Conv2d( 1, hidden_size,  kernel_size=(game_size, 1), stride=(game_size, 1), bias=False,)
        self.im_conv2 = nn.Conv2d(1, 1, kernel_size=(hidden_size, 1), stride=(hidden_size, 1), bias=False)
        self.im_lin2 = nn.Linear(embedding_size * 2 + vocab_size, hidden_size, bias=False)
        self.im_lin3 = nn.Linear(hidden_size, 2, bias=False)
        self.gr_conv1 = GATConv(3, hidden_size, num_heads = head_size)
        self.gr_conv2 = GATConv(hidden_size, embedding_size, num_heads = head_size)
        self.gr_lin1 = nn.Linear(embedding_size*2, embedding_size)

    def forward(self, x, graphs, vocab, _aux_input=None):
        emb = []
        #make sure graphs and images are given in random order
        game_size_list = range(self.game_size)
        order = sorted(game_size_list, key=lambda x: random.random())

        for i in order:
            emb.append(self.return_embeddings(x[i]))

        emb = torch.concat(emb, dim=2)       # batch_size, 1, game_size, embedding_size

        h = self.im_conv1(emb)               # batch_size, hidden_size, 1, embedding_size
        h = torch.nn.LeakyReLU()(h)

        h = h.transpose(1, 2)                # batch_size, 1, hidden_size, embedding_size
        h = self.im_conv2(h)                 # 1, 1, 1, embedding_size
        h = torch.nn.LeakyReLU()(h)

        h = h.squeeze(dim=1)
        h = h.squeeze(dim=1)                 # h of size (batch_size, embedding_size)

        gr_emb = []
        for i in order:
            g = graphs[i]
            feat = g.ndata['object']
            res = self.gr_conv1(g, feat)
            res= res.relu()
            res = self.gr_conv2(g, res)
            g.ndata['h'] = res
            a = dgl.mean_nodes(g, 'h')
            a = dgl.mean_nodes(g, 'h')
            gr_emb.append(a[0][0][0])

        gr_emb = torch.cat(gr_emb, dim=0)
        gr_emb = self.gr_lin1(gr_emb)

        h = torch.cat((gr_emb, h[0]))
        h = torch.cat((h, vocab))
        print(h)
        raise(KeyboardInterrupt)
        h = self.im_lin2(h)
        h = self.im_lin3(h)
        h = h.mul(1.0 / self.temp)
        # h of size (batch_size, vocab_size)
        logits = F.log_softmax(h, dim=0)
        return logits, order.index(0)

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



model = InformedSender(game_size = 2, feat_size = 160000, embedding_size = 30, hidden_size=20, head_size = 2)
model2 = Receiver(game_size = 2, feat_size = 160000, embedding_size = 30, hidden_size=20, head_size = 2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)

acc_list = []

for epoch in range(1000):
    print(epoch)
    images, graphs = get_batch(2)
    pred = model(images, graphs)
    print(pred)
    pred2, label = model2(images, graphs, pred)
    # print("prediction:", pred2, "true label:", label)
    reward = torch.tensor([1.0 if pred2.argmax() == label else 0.0], requires_grad=True)
    # print("reward:", reward)
    
    optimizer.zero_grad()
    reward.backward()
    optimizer.step()
    reward.backward()
    optimizer2.zero_grad()

    optimizer2.step()

    num_correct = 0
    num_tests = 0


for epoch in range(30):
    images, graphs = get_batch(2)

    pred = model(images, graphs)
    pred2, label = model2(images, graphs, pred)
    if pred2.argmax() == label:
        num_correct += 1
    num_tests += 1
    acc_list.append(num_correct / num_tests)

num_correct = 0
num_tests = 0

for epoch in range(100):
    images, graphs = get_batch(2)

    pred = model(images, graphs)
    pred2, label = model2(images, graphs, pred)
    if pred2.argmax() == label:
        num_correct += 1
    num_tests += 1

plt.plot(range(len(acc_list)), acc_list)
plt.show()
print('Test accuracy:', num_correct / num_tests)