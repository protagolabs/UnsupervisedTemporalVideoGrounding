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
from model import Seq2Seq


wordembed_size=300
hidden_size=512
rnn_layers=2
cluster_size=512
dropout=0.5

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    word2idx = json.load(open('./data/dataset/Charades/Charades_word2id.json', 'r'))
    vocab_size = len(word2idx)


    '''
    a=10
    for batchidx, (target_vids,target_qries, lengths_qry,start_frames,end_frames,names, ids) in enumerate(train_dataloader):
        print('target_vids: ', target_vids.shape)
        print('target_qries: ', target_qries.shape)
        print('')
        a=a-1
        if a<=0:
            break
    '''

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = Seq2Seq(vocab_size,wordembed_size,hidden_size,rnn_layers,dropout,device)
    model=model.to(device)


    checkpoint = torch.load('2021-02-26/14_29_48_lan_model.pth.tar')
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)



    necklist1 = {}
    necklist2 = {}
    necklist3 = {}
    necklist4 = {}
    necklistall = {}
    necklist = {}
    testspilt=['book','floor','light','shoes']

    neck_dict = checkpoint['neck_dict']
    for i in range(16):
        necklist1['center{}'.format(i)] = neck_dict[0][i].tolist()
        necklist2['center{}'.format(i)] = neck_dict[1][i].tolist()
        necklist3['center{}'.format(i)] = neck_dict[2][i].tolist()
        necklist4['center{}'.format(i)] = neck_dict[3][i].tolist()

    for split in testspilt:
        test_dataset = CharadesSTA('{}'.format(split))
        test_dataloader = DataLoader(
            test_dataset, batch_size=32,
            shuffle=True, collate_fn=collate_data, num_workers=6, pin_memory=True
        )
        for batchidx, (target_vids, lens_vid, target_qries, lengths_qry, start_frames, end_frames, names, ids) in enumerate(test_dataloader):

            target_qries = target_qries.to(device)
            trg=target_qries.permute(1,0)
            out,neck,inco,deco= model(target_qries, lengths_qry)

            neckout=neck
            neckout=neckout.reshape(neckout.size(0),4,512)


            for i, item in enumerate(neckout):
                necklist1['{}_{}_{}'.format('one',split,names[i],item[0][0])] = item[0].tolist()
                necklist2['{}_{}_{}'.format('two',split,names[i], item[0][0])] = item[1].tolist()
                necklist3['{}_{}_{}'.format('three',split,names[i], item[0][0])] = item[2].tolist()
                necklist4['{}_{}_{}'.format('four',split,names[i], item[0][0])] = item[3].tolist()

    necklist[0] = necklist1
    necklist[1] = necklist2
    necklist[2] = necklist3
    necklist[3] = necklist4

    jsondata = json.dumps(necklist, indent=4, separators=(',', ': '))

    f = open('bfls.json', 'w+')
    f.write(jsondata)
    f.close()


if __name__ == '__main__':
    main()
