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
from model import Interaction,Seq2Seq
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from collections import Counter
from tools import AverageMeter
import copy
from functools import reduce

necktype = 4
space_size = 1024
cluster_size = 512
cluster_type = 16
t1 = 0.0001
t2 = 0.0001
t3 = 0.5
t_trip = 0.2
t_abs = 0.2
weight_abs = 1
wordembed_size = 300
hidden_size = 512
rnn_layers = 2
dropout = 0.5


def cIoU(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return (float(intersection) / union)

def save_checkpoint(state, day, second):
    name = "{}/{}_continuemodel.pth.tar".format(day, second)
    torch.save(state, name)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('2021-02-26/14_29_48_lan_model.pth.tar')
    neck_dict = checkpoint['neck_dict']
    word2idx = json.load(open('./data/dataset/Charades/Charades_word2id.json', 'r'))
    vocab_size = len(word2idx)
    LAN = Seq2Seq(vocab_size, wordembed_size, hidden_size, rnn_layers, dropout, device).to(device)
    pretrained_dict = checkpoint['state_dict']
    model_dict = LAN.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    LAN.load_state_dict(model_dict)


    train_dataset = CharadesSTA('train')

    train_dataloader = DataLoader(
        train_dataset, batch_size=32,
        shuffle=True, collate_fn=collate_data, num_workers=8, pin_memory=True
    )
    test_dataset = CharadesSTA('test')

    test_dataloader = DataLoader(
        test_dataset, batch_size=32,
        shuffle=True, collate_fn=collate_data, num_workers=8, pin_memory=True
    )

    day = time.strftime("%Y-%m-%d", time.localtime())
    if not os.path.isdir(day):
        os.makedirs(day)

    modellist = [Interaction(space_size, cluster_size, cluster_type, device) for _ in range(necktype)]
    savelist = []

    checkpoint = torch.load('2021-03-06/75_tabs0.2_ttrip0.2_flag863model.pth.tar')
    pretrained_dict = checkpoint['state_dict_list']


    for i, dic in enumerate(pretrained_dict):
        print(i)
        model_dict = modellist[i].state_dict()
        pretrained_dict = {k: v for k, v in dic.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        modellist[i].load_state_dict(model_dict)
        savelist.append(modellist[i])
    print(len(savelist))

    savelist = []
    losslist = []

    criterion1 = nn.CrossEntropyLoss()

    for d in range(necktype):
        now_loss = 10000
        model = modellist[d].to(device)
        savelist.append(model)
        learned_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(learned_params, lr=1e-6)
        vidft_dic = {}

        for batchidx, (
                target_vids, lengths_vid, target_qries, lengths_qry, start_frames, end_frames, names,
                ids) in enumerate(
            train_dataloader):
            target_vids = target_vids.to(device)
            spe_ft, spe_att, agn_att, att, H = model(
                torch.FloatTensor(neck_dict[d]).repeat(target_vids.size(0), 1, 1).to(device), target_vids)

            for i, name in enumerate(names):
                if not (name in vidft_dic.keys()):
                    ft = target_vids[i, :lengths_vid[i], ]
                    vidft_dic[name] = torch.mm(agn_att[i, :, :lengths_vid[i]], ft).squeeze(0).tolist()

        vid_ft = torch.FloatTensor(list(vidft_dic.values()))
        vid_ft = F.normalize(vid_ft, p=2)
        estimator = SpectralClustering(n_clusters=cluster_type, affinity='rbf', gamma=1).fit_predict(vid_ft)
        div = {}
        for i, key in enumerate(vidft_dic.keys()):
            div[key] = estimator[i]
        print('agn_score:', metrics.calinski_harabasz_score(vid_ft, estimator))



        for epoch in range(1):
            print(d,epoch,'starting...')
            print('epoch:', epoch, ' starts...')
            losses = AverageMeter()
            losses_cls = AverageMeter()
            losses_abs = AverageMeter()
            losses_trip = AverageMeter()
            for batchidx, (
                    target_vids, lengths_vid, target_qries, lengths_qry, start_frames, end_frames, names,
                    ids) in enumerate(
                train_dataloader):
                target_vids = target_vids.to(device)
                spe_ft, spe_att, agn_att, att, H = model(
                    torch.FloatTensor(neck_dict[d]).repeat(target_vids.size(0), 1, 1).to(device), target_vids)

                div_trg = []

                for i, name in enumerate(names):
                    div_trg.append(div[name])

                loss_cls = criterion1(att, torch.LongTensor(div_trg).to(device)).to(device)
                losses_cls.update(loss_cls)

                loss_inter = 0
                loss_intra = 0

                rd = random.sample(range(0, 15), 4)
                for type in rd:
                    J = []
                    B = []
                    for i, Xcs in enumerate(spe_ft):
                        if div_trg[i] == type:
                            j = torch.mm(Xcs, spe_att[i][type].unsqueeze(-1)).squeeze(-1)
                            J.append(j)
                            att_b = torch.ones_like(spe_att[i][type]) - spe_att[i][type]
                            att_b = att_b / (max(lengths_vid) - 1)
                            b = torch.mm(Xcs, att_b.unsqueeze(-1)).squeeze(-1)
                            B.append(b)
                            '''
                            print('in',target_vids[i])
                            print('out',Xcs)
                            print(b,j)
                            print(lengths_vid[i])
                            print(att_b)
                            print(spe_att[i][type])
                            '''

                    for i, m in enumerate(J):
                        for j, n in enumerate(J):
                            if i != j:
                                loss_inter += max((1 - torch.cosine_similarity(m, n, dim=0)).to(device) - t1, 0)
                                loss_intra += max(torch.cosine_similarity(m, B[i], dim=0).to(device) -
                                                  torch.cosine_similarity(m, n, dim=0).to(device) + t2, 0)
                                '''
                                print('lossinter:',max((1 - torch.cosine_similarity(m, n, dim=0)).to(device) - t1, 0))
                                print('lossintra:',max(torch.cosine_similarity(m, B[i], dim=0).to(device) -
                                                  torch.cosine_similarity(m, n, dim=0).to(device) + t2, 0))
                                print('2:',loss_inter,loss_intra)   
                                print(torch.cosine_similarity(m, B[i], dim=0),torch.cosine_similarity(m, n, dim=0).to(device))
                                print(m,B[i],n)
                                '''

                loss_abs = loss_inter + weight_abs * loss_intra
                losses_abs.update(loss_abs)

                loss_trip = 0
                rd = random.sample(range(0, 15), 4)
                cnt = Counter(div_trg)
                while min([cnt[i] for i in rd]) < 2:
                    rd = random.sample(range(0, 15), 4)

                for type in rd:
                    HN = []
                    HP = []
                    for i, h in enumerate(H):
                        if div_trg[i] == type:
                            HP.append(h)
                        else:
                            HN.append(h)
                    HN = torch.stack(HN)
                    HP = torch.stack(HP)
                    for i, hp in enumerate(HP):
                        dp = 1 - torch.cosine_similarity(hp, HP, dim=-1).sort().values[0]
                        dn = 1 - torch.cosine_similarity(hp, HN, dim=-1).sort().values[-1]
                        loss_trip += max(dp - dn + t3, 0)

                losses_trip.update(loss_trip)

                loss = loss_cls + t_trip * loss_trip + t_abs * loss_abs
                losses.update(loss)
                # backprop

                optimizer.zero_grad()
                if loss != 0:
                    loss.backward()

                optimizer.step()


            losslist.append(losses.avg)
            print('epoch: ', epoch, 'neck: ', d, '\nloss:', losses.avg, '\nloss_cls:', losses_cls.avg, '\nloss_trip:',
                  t_trip * losses_trip.avg, '\nloss_abs:', t_abs * losses_abs.avg)
            vidft_dic.clear()

    copymodel = copy.deepcopy(model)
    savelist.pop()
    savelist.append(copymodel)

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
            spe_ft, spe_att, agn_att, att, H = savelist[d](
                neck[:, d, :].unsqueeze(1).repeat(1, cluster_type, 1), target_vids)
            maxatt = torch.argmax(att, dim=-1)
            pre_att = []
            for i, ma in enumerate(maxatt):
                pre = spe_att[i, ma, :] * agn_att[i]
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

        for i, item in enumerate(sorted):
            total += 1
            floor = np.floor(lengths_vid[i] * 2 / 3) - 1
            if item[-1] < floor:
                (start_seg, end_seg) = (item[-1] * 128, item[-1] * 128 + 128)
            else:
                (start_seg, end_seg) = ((item[-1] - floor) * 256, (item[-1] - floor) * 256 + 256)
            iou = cIoU((start_seg, end_seg), (start_frames[i], end_frames[i]))
            if iou > 0.7:
                iou_7_top_1 += 1
            if iou > 0.5:
                iou_5_top_1 += 1
            if iou > 0.3:
                iou_3_top_1 += 1

        for i, item in enumerate(sorted):
            floor = np.floor(lengths_vid[i] * 2 / 3) - 1
            iou = []
            for j in range(-5, 0):
                j = j % len(item)
                if item[j] < floor:
                    (start_seg, end_seg) = (item[j] * 128, item[j] * 128 + 128)
                else:
                    (start_seg, end_seg) = ((item[j] - floor) * 256, (item[j] - floor) * 256 + 256)
                iou.append(cIoU((start_seg, end_seg), (start_frames[i], end_frames[i])))
            # print(item,iou)
            if max(iou) > 0.7:
                iou_7_top_5 += 1
            if max(iou) > 0.5:
                iou_5_top_5 += 1
            if max(iou) > 0.3:
                iou_3_top_5 += 1

    print('total:', total)
    print('iou_3_top_1:', iou_3_top_1 / total)
    print('iou_5_top_1:', iou_5_top_1 / total)
    print('iou_7_top_1:', iou_7_top_1 / total)
    print('iou_3_top_5:', iou_3_top_5 / total)
    print('iou_5_top_5:', iou_5_top_5 / total)
    print('iou_7_top_5:', iou_7_top_5 / total)


    print(len(savelist),d,epoch,'add to savelist')
    second = time.strftime("%H:%M:%S", time.localtime())
    save_checkpoint({
        'state_dict_list': [model.state_dict() for model in savelist],
        'epoch':epoch,
        'neck':d,
        'loss':losslist
    }, day=day, second=second)


if __name__ == '__main__':
    main()
