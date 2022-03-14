import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import nltk
import json
import scipy.io as sio
import skimage.measure as scikit


class CharadesSTA(Dataset):

    def __init__(self, split='train'):
        self.lang_data = list(open(f"./data/dataset/Charades/Charades_sta_{split}.txt", 'r'))
        self.fps_info = json.load(open('./data/dataset/Charades/Charades_fps_dict.json', 'r'))
        self.duration_info = json.load(open('./data/dataset/Charades/Charades_duration.json', 'r'))
        self.word2id = json.load(open('./data/dataset/Charades/Charades_word2id.json', 'r'))
        self.ft_root = '/home/wyzhen/datasets/C3D/c3d_features/'

        self.name_list = []
        self.query_list = []
        self.time_list = []
        self.frame_list = []
        for item in self.lang_data:
            first_part, query_sentence = item.strip().split('##')
            query_sentence = query_sentence.replace('.', '')
            vid_name, start_time, end_time = first_part.split()
            query_words = nltk.word_tokenize(query_sentence)
            query_tokens = [self.word2id[word] for word in query_words]
            gt_start_time = float(start_time)
            gt_end_time = float(end_time)
            self.name_list.append(vid_name)
            self.query_list.append(query_tokens)
            self.time_list.append((gt_start_time, gt_end_time))

            fps = float(self.fps_info[vid_name])
            gt_start_frame = float(start_time) * fps
            gt_end_frame = float(end_time) * fps
            self.frame_list.append((gt_start_frame, gt_end_frame))

    def __getitem__(self, index):

        id = index

        name = self.name_list[id]
        query = self.query_list[id]
        query = torch.Tensor(query)
        length_qry = len(query)
        start_time, end_time = self.time_list[id]
        start_frame, end_frame = self.frame_list[id]

        video_feat_file = self.ft_root + str(name) + ".mat"
        video_feat_mat = sio.loadmat(video_feat_file)
        video_feat = video_feat_mat['feature']
        # 128 frame features
        video_feat1 = scikit.block_reduce(video_feat, block_size=(8, 1), func=np.mean)
        # 256 frame features
        video_feat2 = scikit.block_reduce(video_feat, block_size=(16, 1), func=np.mean)
        # concatenation of all 128 frame feature and 256 frame feature
        video_feat = np.concatenate((video_feat1, video_feat2), axis=0)

        vid_feature = torch.Tensor(video_feat)

        return vid_feature, query, length_qry, start_frame, end_frame, name, id

    def __len__(self):
        return len(self.name_list)


def collate_data(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    vid_features, queries, lengths_qry, start_frames, end_frames, names, ids = zip(*batch)

    lengths_vid = [len(vid) for vid in vid_features]

    target_vids = torch.zeros(len(vid_features), max(lengths_vid), 4096)

    for i, vid in enumerate(vid_features):
        end = lengths_vid[i]
        target_vids[i, :end, ] = vid[:end, ]

    lengths_qry = [len(cap) for cap in queries]
    lengths_qry = torch.clamp(torch.tensor(lengths_qry), max=10)

    target_qries = torch.zeros(len(queries), 10).long()
    for i, cap in enumerate(queries):
        end = lengths_qry[i]
        target_qries[i, :end] = cap[:end]

    return target_vids, lengths_vid, target_qries, lengths_qry, start_frames, end_frames, names, ids



