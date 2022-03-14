import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import shutil
import torch.nn.functional as F
import yaml
import json
import pickle
import os
import random
from dataset1 import CharadesSTA, collate_data
from tensorboardX import SummaryWriter
import os
import time
from model import Seq2Seq
from sklearn.cluster import KMeans
from tools import AverageMeter

wordembed_size = 300
hidden_size = 512
rnn_layers = 2
cluster_size = 1024
dropout = 0.5
necktype = 2
cluster_type = 16


def save_checkpoint(state, day, second):
    name = "{}/{}_lan_neck2_model.pth.tar".format(day, second)
    torch.save(state, name)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    word2idx = json.load(open('./data/dataset/Charades/Charades_word2id.json', 'r'))
    vocab_size = len(word2idx)
    train_dataset = CharadesSTA('train')

    train_dataloader = DataLoader(
        train_dataset, batch_size=32,
        shuffle=True, collate_fn=collate_data, num_workers=8, pin_memory=True
    )

    kmeans_dataloader = DataLoader(
        train_dataset, batch_size=32,
        shuffle=True, collate_fn=collate_data, num_workers=8, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = Seq2Seq(vocab_size, wordembed_size, hidden_size, rnn_layers, dropout, device)
    model = model.to(device)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    learned_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(learned_params, lr=5e-4)

    day = time.strftime("%Y-%m-%d", time.localtime())
    if not os.path.isdir(day):
        os.makedirs(day)

    now_loss = 10000

    for epoch in range(30):
        print('epoch:',epoch,' starts...')
        losses = AverageMeter()
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses3 = AverageMeter()

        for batchidx, (
        target_vids, lengths_vid, target_qries, lengths_qry, start_frames, end_frames, names, ids) in enumerate(
                train_dataloader):

            target_qries = target_qries.to(device)
            trg = target_qries.permute(1, 0).reshape(-1)
            out, neck, enco, deco = model(target_qries, lengths_qry)
            out = out.reshape(-1, out.shape[-1])

            neck = neck.view(-1, necktype, cluster_size).to(device)
            eyes = 0.5 * torch.eye(necktype, necktype).unsqueeze(0).repeat(neck.size(0), 1, 1).to(device)
            loss3 = torch.norm(torch.bmm(neck, neck.permute(0, 2, 1)) - eyes) / (neck.size(0) * 16)
            loss1 = criterion1(out, trg)
            loss2 = criterion2(enco, deco)
            loss = 10 * loss1 + 10 * loss2 + 10 * loss3
            losses.update(loss)
            losses1.update(loss1)
            losses2.update(loss2)
            losses3.update(loss3)
            # backprop

            optimizer.zero_grad()
            if loss != 0:
                loss.backward()

            optimizer.step()

        print('epoch: ', epoch, '\nloss:', losses.avg, '\nloss1:', 10 * losses1.avg, '\nloss2:', 10 * losses2.avg,
              '\nloss3:', 10 * losses3.avg)

        if(losses.avg < now_loss):
            now_loss = losses.avg
            necklist = [[] for _ in range(necktype)]

            for batchidx, (
            target_vids, lengths_vid, target_qries, lengths_qry, start_frames, end_frames, names, ids) in enumerate(
                    kmeans_dataloader):

                target_qries = target_qries.to(device)
                out, neck, enco, deco = model(target_qries, lengths_qry)

                neck = neck.view(-1, necktype, cluster_size).to(device)
                for i, item in enumerate(neck):
                    for j in range(necktype):
                        necklist[j].append(item[j].tolist())

            clusterdic = {}

            for i in range(necktype):
                estimator = KMeans(n_clusters=cluster_type, max_iter=300, n_init=10).fit(torch.tensor(necklist[i]))
                clusterdic[i] = estimator.cluster_centers_
                print(estimator.score(torch.tensor(necklist[i])))

            second = time.strftime("%H:%M:%S", time.localtime())
            save_checkpoint({
                'state_dict': model.state_dict(),
                'neck_dict': clusterdic
            }, day=day, second=second)
            print(second,' saved!')


if __name__ == '__main__':
    main()
