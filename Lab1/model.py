# -*- coding: UTF-8 -*-
"""
@Project ：SA 
@File ：model.py
@Author ：AnthonyZ
@Date ：2024/10/9 14:50
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout_rate,
        pad_index,
    ):
        super().__init__() 
        # TODO: 定义Embedding https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs)
                for fs in filter_sizes
            ]
        )    
        # TODO：定义线性层 https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#linear 
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids):
        embedded = self.dropout(self.embedding(ids))

        # TODO: embedded = [batch size, seq len, embedding dim] -> [batch size, embedding dim, seq len] 将获得的embedding转换为正确的维度 https://pytorch.org/docs/stable/generated/torch.permute.html#torch.permute
        embedded = embedded.permute(0, 2, 1)

        # TODO：使用定义的卷积层对文本特征进行提取，卷积层之间需要加入激活函数 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#conv1d https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#relu
        conved = [F.relu(conv(embedded)) for conv in self.convs]

        # TODO：对提取的特征进行最大池化
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=-1))

        # TODO：利用线性层，将获得的特征信息进行分类 https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#linear
        prediction = self.fc(cat)

        return prediction



class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_rate, pad_index, pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()
        # 如果提供了预训练的嵌入矩阵，则使用它，否则随机初始化
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=pad_index, freeze=freeze_embeddings)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids):
        embedded = self.dropout(self.embedding(ids))
        output, (hidden, cell) = self.lstm(embedded)
        prediction = self.fc(hidden[-1])
        return prediction