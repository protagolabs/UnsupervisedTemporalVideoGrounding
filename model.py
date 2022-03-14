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
from tensorboardX import SummaryWriter
import os
import time

class Seq2Seq(nn.Module):
    class Encoder(nn.Module):

        def __init__(self, vocab_size,wordembed_size,hidden_size,rnn_layers,dropout,device):
            super(Seq2Seq.Encoder, self).__init__()
            self.embedding = nn.Embedding(vocab_size + 1, wordembed_size, padding_idx=0).to(device)
            self.embedding.weight.data.copy_(torch.load('./data/glove_weights'))
            self.embedding.requires_grad = False
            self.rnn = nn.LSTM(wordembed_size, hidden_size, num_layers=rnn_layers, dropout=dropout)

            self.dropout = nn.Dropout(dropout)

        def forward(self, src):
            # src = [src sent len, batch size] 这句话的长度和batch大小
            embedded = self.dropout(self.embedding(src))
            # embedded = [src sent len, batch size, emb dim]
            output, (hidden, cell) = self.rnn.forward(embedded)
            # x = [scr len, emb_dim]
            # w_xh = [emb_dim, hid_dim, n_layers]
            # scr sen len, batch size, hid dim, n directions, n layers
            # outputs: [src sent len, batch size, hid dim * n directions]
            # hidden, cell: [n layers* n directions, batch size, hid dim]
            # outputs are always from the top hidden layer
            # The RNN returns:
            # outputs (the top-layer hidden state for each time-step)
            # hidden (the final hidden state for each layer, stacked on top of
            # each other)
            return hidden, cell
    class Decoder(nn.Module):

        def __init__(self, vocab_size,wordembed_size,hidden_size,rnn_layers,dropout,device):
            super(Seq2Seq.Decoder, self).__init__()
            self.wordembed_size=wordembed_size
            self.device=device
            self.embedding = nn.Embedding(vocab_size + 1, wordembed_size, padding_idx=0).to(device)
            self.embedding.weight.data.copy_(torch.load('./data/glove_weights'))
            self.embedding.requires_grad = False
            self.rnn = nn.LSTM(wordembed_size, hidden_size, rnn_layers, dropout=dropout)
            self.out = nn.Linear(hidden_size, vocab_size + 1)
            self.dropout = nn.Dropout(dropout)
            self.vocab_size = vocab_size + 1

        def forward(self, input, hidden, cell,t):
            # input = [batch size]
            # hidden = [n layers * n directions, batch size, hid dim]
            # cell = [n layers * n directions, batch size, hid dim]
            # n directions in the decoder will both always be 1, therefore:
            # hidden = [n layers, batch size, hid dim]
            # context = [n layers, batch size, hid dim]

            if t == 0:
                start=torch.ones(1,input.size(1),self.wordembed_size).to(self.device)
                start=start/100
                embedded = self.dropout(start)
            else:
                input = input.unsqueeze(0)
                # input = [1, batch size]
                embedded = self.dropout(self.embedding(input))
                # embedded = [1, batch size, emb dim]
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

            output_new = output.squeeze(0)
            # output_new = [batch size, hid dim]
            prediction = self.out(output_new)
            # prediction = [batch size, output dim]
            return prediction, hidden, cell

    def __init__(self,vocab_size,wordembed_size,hidden_size,rnn_layers,dropout, device):

        super(Seq2Seq, self).__init__()
        self.encoder = self.Encoder(vocab_size,wordembed_size,hidden_size,rnn_layers,dropout,device)
        self.decoder = self.Decoder(vocab_size,wordembed_size,hidden_size,rnn_layers,dropout,device)
        self.device = device
        self.hidden_size=hidden_size
        self.en = nn.Sequential()
        self.ln1 = nn.Linear(hidden_size*4,hidden_size*4)
        bound = np.sqrt(6. / (hidden_size*8))
        nn.init.uniform_(self.ln1.weight, -bound, bound)
        self.en.add_module('linear1',self.ln1)
        self.en.add_module('Tanh1', nn.Tanh())

        self.de = nn.Sequential()

        self.ln3 = nn.Linear(hidden_size*4,hidden_size*4)
        bound = np.sqrt(6. / (hidden_size*8))
        nn.init.uniform_(self.ln3.weight, -bound, bound)
        self.de.add_module('linear3',self.ln3)

    def forward(self, target_qries,lengths_qry, teacher_forcing_ratio=0.5):

        # src = [src sent len, batch size]

        # trg = [trg sent len, batch size]

        # teacher_forcing_ratio is probability to use teacher forcing

        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        src=target_qries.permute(1,0)
        trg=target_qries.permute(1,0)
        batch_size = trg.shape[1]
        max_len = trg.shape[0]

        trg_vocab_size = self.decoder.vocab_size



        # tensor to store decoder outputs

        outputs = torch.zeros(max_len, batch_size,

                              trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder.forward(src)

        nh = hidden.permute(1,0,2)
        nc = cell.permute(1,0,2)
        inco = torch.cat([nh,nc],2)
        inco = inco.reshape(inco.size(0),-1)
        neck = self.en(inco)
        deco = self.de(neck)
        ta=deco.reshape(inco.size(0),4,self.hidden_size)
        hidden, cell = torch.split(ta,2,dim=1)
        hidden = hidden.permute(1,0,2).contiguous()
        cell = cell.permute(1,0,2).contiguous()


        # first input to the decoder is the <sos> tokens

        input = trg

        for t in range(0, max_len):
            output, hidden, cell = self.decoder.forward(input, hidden, cell,t)
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs,neck,inco,deco


class Interaction(nn.Module):
    def __init__(self,space_size,cluster_size,cluster_type,device):
        super(Interaction, self).__init__()
        self.device = device
        self.space_size=space_size
        self.local = nn.Conv1d(4096,4096,kernel_size=1,padding=0,bias=False)
        nn.init.kaiming_uniform_(self.local.weight, a=1)

        self.agnosic = nn.Conv1d(4096,4096,kernel_size=1,padding=0,bias=False)
        nn.init.kaiming_uniform_(self.agnosic.weight, a=1)

        self.agnatt = nn.Conv1d(4096, 1, kernel_size=1, padding=0,bias=False)
        nn.init.kaiming_uniform_(self.agnatt.weight, a=1)

        self.specific = nn.Conv1d(4096, 4096, kernel_size=1, padding=0,bias=False)
        nn.init.kaiming_uniform_(self.specific.weight, a=1)

        self.VideoToSpace = nn.Conv1d(4096,space_size,kernel_size=3,padding=1,bias=False)
        nn.init.kaiming_uniform_(self.VideoToSpace.weight, a=1)

        self.TextToSpace = nn.Linear(cluster_size,space_size)
        bound = np.sqrt(6. / (cluster_size+space_size))
        nn.init.uniform_(self.TextToSpace.weight, -bound, bound)

        self.last = nn.Linear(cluster_type,cluster_type)
        bound = np.sqrt(6. / (cluster_type+cluster_type))
        nn.init.uniform_(self.last.weight, -bound, bound)

        self.ac = nn.Tanh()

        self.bn = nn.BatchNorm1d(4096)


    def forward(self,clusters,video_features):
        '''
        clusters batch,type,512
        video_features batch,T,4096
        '''
        video_features = video_features.permute(0, 2, 1)
        video_features = self.local(video_features)
        ang_ft = self.agnosic(video_features)#batch,4096,T
        ang_att = self.agnatt(self.bn(self.ac(ang_ft)))#batch,1,T

        ClusterInSpace = []
        for i,cluster in enumerate(clusters):
            ClusterInSpace.append(self.TextToSpace(cluster))

        ClusterInSpace = torch.stack(ClusterInSpace)
        spe_ft = self.specific(video_features) #batch,4096,T
        VideoInSpace = self.VideoToSpace(self.bn(self.ac(spe_ft)))
        spe_att = torch.bmm(ClusterInSpace,VideoInSpace)

        spe_att = F.softmax(spe_att, dim=2) #batch,type,T

        att = torch.bmm(spe_att,ang_att.permute(0,2,1)).squeeze(-1) #batch,type

        att = self.last(self.ac(att))

        H = torch.bmm(ang_ft,ang_att.permute(0,2,1)).squeeze(-1) #batch,4096

        return spe_ft,spe_att,ang_att,att,H







