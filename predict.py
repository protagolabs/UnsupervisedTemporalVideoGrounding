import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import shutil
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
import pickle
import os
import random
from dataset2 import CharadesSTA, collate_data
from tensorboardX import SummaryWriter
import os
import time
from model import Interaction, Seq2Seq
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from functools import reduce

wordembed_size = 300
hidden_size = 512
rnn_layers = 2
cluster_size = 512
dropout = 0.5
necktype = 4
space_size = 1024
cluster_type = 16
t1 = 0.0001
t2 = 0.0001
t3 = 0.5
t_trip = 0.5
t_abs = 0.5
weight_abs = 1
th = 99999


def cIoU(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return (float(intersection) / union)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word2idx = json.load(open('./data/dataset/Charades/Charades_word2id.json', 'r'))
    vocab_size = len(word2idx)
    LAN = Seq2Seq(vocab_size, wordembed_size, hidden_size, rnn_layers, dropout, device).to(device)
    checkpoint = torch.load('2021-02-26/14_29_48_lan_model.pth.tar')
    pretrained_dict = checkpoint['state_dict']
    model_dict = LAN.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    LAN.load_state_dict(model_dict)

    checkpoint = torch.load('2021-03-12/31_tabs0.5_ttrip0.5_flag684model.pth.tar')
    interaction = [Interaction(space_size, cluster_size, cluster_type, device).to(device) for _ in range(necktype)]
    pretrained_dict = checkpoint['state_dict_list']
    print(len(pretrained_dict))
    for i, dic in enumerate(pretrained_dict):
        model_dict = interaction[i].state_dict()
        pretrained_dict = {k: v for k, v in dic.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        interaction[i].load_state_dict(model_dict)

    test_dataset = CharadesSTA('test')
    test_dataloader = DataLoader(
        test_dataset, batch_size=32,
        shuffle=True, collate_fn=collate_data, num_workers=8, pin_memory=True
    )
    total = 0
    iou_3_top_1 = 0
    iou_5_top_1 = 0
    iou_7_top_1 = 0
    iou_3_top_5 = 0
    iou_5_top_5 = 0
    iou_7_top_5 = 0

    for batchidx, (
            target_vids, lengths_vid, target_qries, lengths_qry, start_frames, end_frames, names,
            ids) in enumerate(
        test_dataloader):
        att_list = []
        for d in range(necktype):
            target_qries = target_qries.to(device)
            out, neck, enco, deco = LAN(target_qries, lengths_qry)
            neck = neck.view(-1, necktype, cluster_size).to(device)
            target_vids = target_vids.to(device)
            spe_ft, spe_att, agn_att, att, H = interaction[d](neck[:, d, :].unsqueeze(1).repeat(1,cluster_type,1), target_vids)
            maxatt = torch.argmax(att,dim=-1)
            pre_att = []
            for i,ma in enumerate(maxatt):
                pre = spe_att[i,ma,:] * agn_att[i]
                pre = pre.squeeze(1)
                pre = F.softmax(pre, dim=-1)
                pre_att.append(pre)
            pre_att = torch.stack(pre_att)
            att_list.append(pre_att)
        att = reduce(lambda x, y: x * y, att_list)
        sort = att.sort()[1]
        sorted = []
        for i, item in enumerate(sort):
                sorted.append([rank for rank in item[0] if rank < lengths_vid[i]])

        '''
        for i, item in enumerate(sorted):
            total += 1
            floor = np.floor(lengths_vid[i] * 2 / 3) - 1
            iou = 0
            for j in range(30):
                if j < 16:
                    (start_seg, end_seg) = (j * 128, j * 128 + 128)
                elif j < 24:
                    (start_seg, end_seg) = ((j-16) * 256, (j-16 +1 ) * 256)
                else:
                    (start_seg, end_seg) = ((j - 24) * 512, (j - 24 + 1) * 512)
                if cIoU((start_seg, end_seg),(start_frames[i], end_frames[i])) > iou:
                    iou = cIoU((start_seg, end_seg),(start_frames[i], end_frames[i]))
            if iou > 0.7:
                iou_7_top_1 += 1
            if iou > 0.5:
                iou_5_top_1 += 1
            if iou > 0.3:
                iou_3_top_1 += 1
        '''
        for i, item in enumerate(sorted):
            total += 1
            floor = np.floor(lengths_vid[i] * 2 / 3) - 1
            iou=[]
            for j in range(-1,0):
                j = j % len(item)

                if item[j]< floor:

                    start =  item[j]
                    while start >0:
                        if att[i][0][start-1]/att[i][0][item[j]] >th:
                            start = start -1
                        else:
                            break

                    end = item[j]
                    while end < floor-1:
                        if att[i][0][end+1]/att[i][0][item[j]] >th:
                            end = end + 1
                        else:
                            break
                    (start_seg, end_seg) = (item[j] * 128, item[j] * 128 + 128)
                else:
                    start = item[j]
                    while start > floor:
                        if att[i][0][start-1] / att[i][0][item[j]] > th:
                            start = start - 1
                        else:
                            break
                    end = item[j]
                    while end < len(item)-1:
                        if att[i][0][end+1] / att[i][0][item[j]] > th:
                            end = end + 1
                        else:
                            break
                    (start_seg, end_seg) = ((start-floor) * 256, (end-floor) * 256 + 256)
                iou.append(cIoU((start_seg, end_seg),(start_frames[i], end_frames[i])))
                print(names[i], cIoU((start_seg, end_seg),(start_frames[i], end_frames[i])), (start_seg, end_seg), (start_frames[i], end_frames[i]))

            if max(iou) > 0.7:
                iou_7_top_1 += 1
            if max(iou) > 0.5:
                iou_5_top_1 += 1
            if max(iou) > 0.3:
                iou_3_top_1 += 1
        '''
        print('iou_3_top_1:', iou_3_top_1)
        print('iou_5_top_1:', iou_5_top_1)
        print('iou_7_top_1:', iou_7_top_1)
        
        for i, item in enumerate(sorted):
            floor = np.floor(lengths_vid[i] * 2 / 3) - 1
            iou = []
            for j in range(-5, 0):
                j = j % len(item)

                if item[j] < floor:

                    start = item[j]
                    while start > 0:
                        if att[i][0][start - 1] / att[i][0][item[j]] > th:
                            start = start - 1
                        else:
                            break

                    end = item[j]
                    while end < floor - 1:
                        if att[i][0][end + 1] / att[i][0][item[j]] > th:
                            end = end + 1
                        else:
                            break
                    (start_seg, end_seg) = (item[j] * 128, item[j] * 128 + 128)
                else:
                    start = item[j]
                    while start > floor:
                        if att[i][0][start - 1] / att[i][0][item[j]] > th:
                            start = start - 1
                        else:
                            break
                    end = item[j]
                    while end < len(item) - 1:
                        if att[i][0][end + 1] / att[i][0][item[j]] > th:
                            end = end + 1
                        else:
                            break
                    (start_seg, end_seg) = ((start - floor) * 256, (end - floor) * 256 + 256)
                iou.append(cIoU((start_seg, end_seg), (start_frames[i], end_frames[i])))
            if max(iou) > 0.7:
                iou_7_top_5 += 1
            if max(iou) > 0.5:
                iou_5_top_5 += 1
            if max(iou) > 0.3:
                iou_3_top_5 += 1

        print('total:', total)
        print('iou_3_top_1:', iou_3_top_1)
        print('iou_5_top_1:', iou_5_top_1)
        print('iou_7_top_1:', iou_7_top_1)
        print('iou_3_top_5:', iou_3_top_5)
        print('iou_5_top_5:', iou_5_top_5)
        print('iou_7_top_5:', iou_7_top_5)

        for i, item in enumerate(sorted):
            floor = np.floor(lengths_vid[i] * 2 / 3) - 1
            iou=[]
            for _ in range(-5,0):
                j = random.randint(0,200)
                j = j % len(item)
                if item[j]< floor:
                    (start_seg, end_seg) = (item[j] * 128, item[j] * 128 + 128)
                else:
                    (start_seg, end_seg) = ((item[j]-floor) * 256, (item[j]-floor) * 256 + 256)
                iou.append(cIoU((start_seg, end_seg),(start_frames[i], end_frames[i])))
            if max(iou) > 0.7:
                iou_7_top_5 += 1
            if max(iou) > 0.5:
                iou_5_top_5 += 1
            if max(iou) > 0.3:
                iou_3_top_5 += 1

        for i, item in enumerate(sorted):
            floor = np.floor(lengths_vid[i] * 2 / 3) - 1
            iou=[]
            for _ in range(-10,0):
                j = random.randint(0, 200)
                j = j % len(item)
                if item[j] < floor:
                    (start_seg, end_seg) = (item[j] * 128, item[j] * 128 + 128)
                else:
                    (start_seg, end_seg) = ((item[j]-floor) * 256, (item[j]-floor) * 256 + 256)
                iou.append(cIoU((start_seg, end_seg),(start_frames[i], end_frames[i])))
                #print(cIoU((start_seg, end_seg),(start_frames[i], end_frames[i])),(start_seg.item(), end_seg.item()),(start_frames[i], end_frames[i]))
            if max(iou) > 0.7:
                iou_7_top_10 += 1
            if max(iou) > 0.5:
                iou_5_top_10 += 1
            if max(iou) > 0.3:
                iou_3_top_10 += 1
        print('total:', total)
        print('iou_3_top_1:', iou_3_top_1)
        print('iou_5_top_1:', iou_5_top_1)
        print('iou_7_top_1:', iou_7_top_1)
        print('iou_3_top_5:', iou_3_top_5)
        print('iou_5_top_5:', iou_5_top_5)
        print('iou_7_top_5:', iou_7_top_5)
        print('iou_3_top_10:', iou_3_top_10)
        print('iou_5_top_10:', iou_5_top_10)
        print('iou_7_top_10:', iou_7_top_10)

        for i, item in enumerate(sorted):
            floor = np.floor(lengths_vid[i] * 2 / 3) - 1
            iou=[]
            for j in range(len(item)):
                if item[j] < floor:
                    (start_seg, end_seg) = (item[j] * 128, item[j] * 128 + 128)
                else:
                    (start_seg, end_seg) = (item[j] * 256, item[j] * 128 + 256)
                iou.append(float(cIoU((start_seg, end_seg), (start_frames[i], end_frames[i]))))
            print(names[i],lengths_vid[i])
            print([i.item() for i in item])
            print(iou)
        '''

    print('total:',total)
    print('iou_3_top_1:',iou_3_top_1/total)
    print('iou_5_top_1:',iou_5_top_1/total)
    print('iou_7_top_1:',iou_7_top_1/total)
    print('iou_3_top_5:',iou_3_top_5/total)
    print('iou_5_top_5:',iou_5_top_5/total)
    print('iou_7_top_5:',iou_7_top_5/total)


if __name__ == '__main__':
    main()
