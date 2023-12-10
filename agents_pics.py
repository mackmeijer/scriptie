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

#supports 1 , standson 2 , underneath 3, hangsabove 4, hangsunder 5
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

    return new_im, label

def get_batch(batch_size):
    x = []
    x_labels = []
    for i in range(batch_size):
        g, label = pic_gen()
        x.append(g)
        x_labels.append(label)

    x_labels = torch.concat(x_labels)
    return x, x_labels



class InformedSender(nn.Module):
    def __init__(
        self, game_size, feat_size, embedding_size, hidden_size, vocab_size=100, temp=1.0,
    ):
        super(InformedSender, self).__init__()
        self.game_size = 2
        self.embedding_size = 30
        self.hidden_size = 80
        self.vocab_size = vocab_size
        self.temp = temp

        self.lin1 = nn.Linear(feat_size, embedding_size, bias=False)
        self.conv2 = nn.Conv2d( 1, hidden_size,  kernel_size=(game_size, 1), stride=(game_size, 1), bias=False,)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=(hidden_size, 1), stride=(hidden_size, 1), bias=False)
        self.lin4 = nn.Linear(embedding_size, vocab_size, bias=False)

  
    def forward(self, x, _aux_input=None):
        emb = self.return_embeddings(x)
        print("wa", emb.size())
        # in: h of size (batch_size, 1, game_size, embedding_size)
        # out: h of size (batch_size, hidden_size, 1, embedding_size)
        h = self.conv2(emb)
        h = torch.sigmoid(h)
        # in: h of size (batch_size, hidden_size, 1, embedding_size)
        # out: h of size (batch_size, 1, hidden_size, embedding_size)
        h = h.transpose(1, 2)
        h = self.conv3(h)
        # h of size (batch_size, 1, 1, embedding_size)
        h = torch.sigmoid(h)
        h = h.squeeze(dim=1)
        h = h.squeeze(dim=1)
        # h of size (batch_size, embedding_size)
        h = self.lin4(h)
        h = h.mul(1.0 / self.temp)
        # h of size (batch_size, vocab_size)
        logits = F.log_softmax(h, dim=1)

        return logits

    def return_embeddings(self, x):
        # Embed each image (left or right)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            h = torch.from_numpy(h).float()
            
            # Assuming h has a size of [400 x 400], you may want to flatten it
            h = h.view(-1)
            
            # Apply the linear layer
            h_i = self.lin1(h)
            
            # Reshape the result to [batch_size x 1 x 1 x embedding_size]
            h_i = h_i.view(-1, 1, 1, self.embedding_size)
            
            # Append to the list of embeddings
            embs.append(h_i)
        
        # Concatenate the embeddings along the fourth dimension
        h = torch.cat(embs, dim=2)
        print(h.size())
        return h






class Receiver(nn.Module):
    def __init__(self, game_size, feat_size, embedding_size, vocab_size, reinforce):
        super(Receiver, self).__init__()
        self.game_size = game_size
        self.embedding_size = embedding_size

        self.lin1 = nn.Linear(feat_size, embedding_size, bias=False)
        if reinforce:
            self.lin2 = nn.Embedding(vocab_size, embedding_size)
        else:
            self.lin2 = nn.Linear(vocab_size, embedding_size, bias=False)

    def forward(self, signal, x, _aux_input=None):
        # embed each image (left or right)
        emb = self.return_embeddings(x)
        # embed the signal
        if len(signal.size()) == 3:
            signal = signal.squeeze(dim=-1)
        h_s = self.lin2(signal)
        # h_s is of size batch_size x embedding_size
        h_s = h_s.unsqueeze(dim=1)
        # h_s is of size batch_size x 1 x embedding_size
        h_s = h_s.transpose(1, 2)
        # h_s is of size batch_size x embedding_size x 1
        out = torch.bmm(emb, h_s)
        # out is of size batch_size x game_size x 1
        out = out.squeeze(dim=-1)
        # out is of size batch_size x game_size
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


    def return_embeddings(self, x):
        # Embed each image (left or right)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            h = torch.from_numpy(h).float()
            
            # Assuming h has a size of [400 x 400], you may want to flatten it
            h = h.view(-1)
            
            # Apply the linear layer
            h_i = self.lin1(h)
            
            # Reshape the result to [batch_size x 1 x 1 x embedding_size]
            h_i = h_i.view(-1, 1, 1, self.embedding_size)
            
            # Append to the list of embeddings
            embs.append(h_i)
        
        # Concatenate the embeddings along the fourth dimension
        h = torch.cat(embs, dim=2)
        print(h.size())
        return h





model = InformedSender(game_size = 2, feat_size = 160000, embedding_size = 30, hidden_size=80)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

acc_list = []

for epoch in range(100):
    images, labels = get_batch(2)
    pred = model(images)
    print(pred)
    loss = F.cross_entropy(pred, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


num_correct = 0
num_tests = 0
for epoch in range(10):
    image, label = pic_gen()
    image.save('test.jpg')
    im_gray = cv2.imread('test2.jpg', cv2.IMREAD_GRAYSCALE)
    pred = model(im_gray)
    if pred.argmax() == label:
        num_correct += 1
    num_tests += 1

plt.plot(range(len(acc_list)), acc_list)
plt.show()
print('Test accuracy:', num_correct / num_tests)




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="")
    parser.add_argument( "--tau_s", type=float, default=10.0)
    parser.add_argument( "--game_size", type=int, default=2)
    parser.add_argument("--same", type=int, default=0)
    parser.add_argument("--embedding_size", type=int, default=50)
    parser.add_argument( "--hidden_size",  type=int,  default=20)
    parser.add_argument( "--batches_per_epoch", type=int,  default=100)
    parser.add_argument("--inf_rec", type=int, default=0)
    parser.add_argument( "--mode",  type=str, default="rf")
    parser.add_argument("--gs_tau", type=float, default=1.0)

    opt = core.init(parser)
    assert opt.game_size >= 1

    return opt


def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    """
    Accuracy loss - non-differetiable hence cannot be used with GS
    """
    acc = (labels == receiver_output).float()
    return -acc, {"acc": acc}


def loss_nll(
    _sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
):
    """
    NLL loss - differentiable and can be used with both GS and Reinforce
    """
    nll = F.nll_loss(receiver_output, labels, reduction="none")
    acc = (labels == receiver_output.argmax(dim=1)).float().mean()
    return nll, {"acc": acc}


def get_game(opt):
    feat_size = 160000
    sender = InformedSender(
        opt.game_size,
        feat_size,
        opt.embedding_size,
        opt.hidden_size,
        opt.vocab_size,
        temp=opt.tau_s,
    )
    receiver = Receiver(
        opt.game_size,
        feat_size,
        opt.embedding_size,
        opt.vocab_size,
        reinforce=(opts.mode == "rf"),
    )
    if opts.mode == "rf":
        sender = core.ReinforceWrapper(sender)
        receiver = core.ReinforceWrapper(receiver)
        game = core.SymbolGameReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=0.01,
            receiver_entropy_coeff=0.01,
        )
    elif opts.mode == "gs":
        sender = core.GumbelSoftmaxWrapper(sender, temperature=opt.gs_tau)
        game = core.SymbolGameGS(sender, receiver, loss_nll)
    else:
        raise RuntimeError(f"Unknown training mode: {opts.mode}")

    return game



# opts = parse_arguments()
# game = get_game(opts)
# optimizer = core.build_optimizer(game.parameters())
# callback = None

# if opts.mode == "gs":
#     callbacks = [core.TemperatureUpdater(agent=game.sender, decay=0.9, minimum=0.1)]
# else:
#     callbacks = []

# callbacks.append(core.ConsoleLogger(as_json=True, print_train_loss=True))
# trainer = core.Trainer(
#     game=game,
#     optimizer=optimizer,
#     train_data= get_batch(100),
#     validation_data= get_batch(20),
#     callbacks=callbacks,
# )
# print(opts.n_epochs)
# trainer.train(n_epochs=opts.n_epochs)

# core.close()


# model = InformedSender(hidden_size= 4, vocab_size= 20, head_size= 2)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# acc_list = []


# for epoch in range(100):
#     batched_graph, labels = get_batch(5)
#     for i in range(5):
#         pred = model(batched_graph[i], batched_graph[i].ndata['object'])
#         loss = F.cross_entropy(pred, labels[i])
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     batched_graph, labels = get_batch(30)
#     num_correct = 0
#     num_tests = 0
#     for i in range(30):
#         pred = model(batched_graph[i], batched_graph[i].ndata['object'])
#         if pred.argmax() == labels[i]:
#             num_correct += 1
#         num_tests += 1
#     acc_list.append(num_correct / num_tests)

# num_correct = 0
# num_tests = 0
# for epoch in range(10):
#     batched_graph, labels = get_batch(5)
#     for i in range(5):
#         pred = model(batched_graph[i], batched_graph[i].ndata['object'])
#         if pred.argmax() == labels[i]:
#             num_correct += 1
#         num_tests += 1

# plt.plot(range(len(acc_list)), acc_list)
# plt.show()
# print('Test accuracy:', num_correct / num_tests)
