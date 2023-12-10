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

#supports 1 , standson 2 , underneath 3, hangsabove 4, hangsunder 5

templates = [[0, 1], [2, 0], [2, 1], [2, 2]]
images = [[r".\img\fles.png", r".\img\lama.png", r".\img\tulp.png"], [r".\img\pilaar.png", r".\img\tafel.png", r".\img\stoel.png"], [r".\img\leeg.png", r".\img\leeg.png", r".\img\schilderij.png"]]

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

#zit_op 0, ondersteund 1, hangt boven 2, hangt onder 3, staat onder 4, links 5, rechts 6




g = dgl.graph(([0, 1,  2,  3,  0,  1, 2,  3],
               [1, 0,  3,  2,  2,  3, 0,  1]), num_nodes=4)
print(g.edges())


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
print(g.edata['relation'])

print(g)
if g.edata['relation'][0][0] == 1 or g.edata['relation'][1][0]:
    label = torch.tensor(1)
else:
    label = torch.tensor(0)

graph = (g, label) 
print(graph)


