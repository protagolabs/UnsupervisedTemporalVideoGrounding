import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import os
from sklearn import manifold
from sklearn.cluster import KMeans


model = json.load(open("bfls.json", 'r'))

X = []
y = []


for key,value in model['0'].items():
    X.append(value)
    y.append(key)

X=np.array(X)
y=np.array(y)
'''t-SNE'''
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)
print(X.shape)
print(X_tsne.shape)
print(y.shape)
print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(8, 8))
walk=0
run=0
eat=0
drink=0
center = 0
read = 0
look = 0
for i in range(X_norm.shape[0]):
    if 'book' in y[i]:
        color = 'green'
        run += 1
    elif 'floor' in y[i]:
        color = 'yellow'
        walk += 1
    elif 'light' in y[i]:
        color = 'blue'
        eat += 1
    elif 'shoes' in y[i]:
        color = 'red'
        drink += 1

    else:
        for c in [6,7,4,11]:
            if 'center{}'.format(c) in y[i]:
                print(y[i])
                color = 'black'
                center += 1
                plt.text(X_norm[i, 0], X_norm[i, 1], '#', color=color,
                         fontdict={'weight': 'bold', 'size': 30})
                continue

            else:
                color = 'white'
                continue
    plt.text(X_norm[i, 0], X_norm[i, 1], '*', color=color,
             fontdict={'weight': 'bold', 'size': 15})
plt.xticks([])
plt.yticks([])
plt.show()
plt.savefig('test.png', dpi = 600)
print('ok','/nglass:',walk,'/ncup:',run,'/nwindow:',drink,'/ndoor:',eat ,'/nread:',read,'/nlook:',look,'/ncenter:',center)
