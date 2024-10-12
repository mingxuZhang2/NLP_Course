# -*- coding: UTF-8 -*-
"""
@Project ：SA 
@File ：main.py
@Author ：AnthonyZ
@Date ：2024/10/9 13:26
"""
from model import CNN, LSTM
from utils import *
from data import *
import matplotlib.pyplot as plt

import time

import argparse
import collections
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import torchtext


def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


def train(data_loader, model, criterion, optimizer, device):
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


if __name__ == '__main__':
    train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])

    tokenizer = basic_english_normalize
    # TODO：超参数，可自行调整
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", default=256)
    parser.add_argument("--test_size", default=0.25)
    parser.add_argument("--min_freq", default=5) 
    parser.add_argument("--batch_size", default=256)
    parser.add_argument("--embedding_dim", default=300)
    parser.add_argument("--n_filters", default=100)
    parser.add_argument("--filter_sizes", default=[3, 5, 7])
    parser.add_argument("--dropout_rate", default=0.3)
    parser.add_argument("--n_epochs", default=10)
    parser.add_argument("--device", default="cuda:5")
    args = parser.parse_args()

    train_data = train_data.map(
        tokenize_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": args.max_length}
    )
    test_data = test_data.map(
        tokenize_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": args.max_length}
    )

    train_valid_data = train_data.train_test_split(test_size=args.test_size)
    train_data = train_valid_data["train"]
    valid_data = train_valid_data["test"]

    special_tokens = ["<unk>", "<pad>"]

    vocab = torchtext.vocab.build_vocab_from_iterator(
        train_data["tokens"],
        min_freq=args.min_freq,
        specials=special_tokens,
    )

    unk_index = vocab["<unk>"]
    pad_index = vocab["<pad>"]

    vocab.set_default_index(unk_index)

    train_data = train_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
    valid_data = valid_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
    test_data = test_data.map(numericalize_example, fn_kwargs={"vocab": vocab})

    train_data = train_data.with_format(type="torch", columns=["ids", "label"])
    valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
    test_data = test_data.with_format(type="torch", columns=["ids", "label"])

    train_data_loader = get_data_loader(train_data, args.batch_size, pad_index, shuffle=True)
    valid_data_loader = get_data_loader(valid_data, args.batch_size, pad_index)
    test_data_loader = get_data_loader(test_data, args.batch_size, pad_index)

    vocab_size = len(vocab)
    output_dim = len(train_data.unique("label"))
    # 探究不同超参数对模型性能的影响，embedding_dim, n_filters, filter_sizes, dropout_rate 

    embedding_dim = [200, 250, 300]
    n_filters = [100, 150, 200]
    filter_sizes = [[2, 4, 6], [4, 6, 8], [5, 7, 9]]
    dropout_rate = [0.3]

    # record all of the results with different hyper-parameters
    results = []

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(1):
                    model = CNN(
                        vocab_size=vocab_size,
                        embedding_dim=embedding_dim[i],
                        n_filters=n_filters[j],
                        filter_sizes=filter_sizes[k],
                        output_dim=output_dim,
                        dropout_rate=dropout_rate[l],
                        pad_index=pad_index,
                    )

                    optimizer = optim.Adam(model.parameters(), lr=0.0005)

                    criterion = nn.CrossEntropyLoss()

                    metrics = collections.defaultdict(list)

                    model = model.to(args.device)

                    criterion = criterion.to(args.device)

                    metrics_CNN = collections.defaultdict(list)

                    print("Now CNN performance:\n")

                    for epoch in range(args.n_epochs):
                        train_loss, train_acc = train(train_data_loader, model, criterion, optimizer, args.device)
                        valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, args.device)
                        metrics_CNN["train_losses"].append(train_loss)
                        metrics_CNN["train_accs"].append(train_acc)
                        metrics_CNN["valid_losses"].append(valid_loss)
                        metrics_CNN["valid_accs"].append(valid_acc)

                        if not metrics_CNN["valid_accs"] or valid_acc > max(metrics_CNN["valid_accs"]):
                            torch.save(model.state_dict(), "best_model_CNN.pth")
                            best_valid_acc = valid_acc

                        print(f"epoch: {epoch}")
                        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
                        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")
                    
                    # testing
                    test_loss, test_acc = evaluate(test_data_loader, model, criterion, args.device)
                    print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")
                    #record the results

                    results.append({
                        "embedding_dim": embedding_dim[i],
                        "n_filters": n_filters[j],
                        "filter_sizes": filter_sizes[k],    
                        "dropout_rate": dropout_rate[l],
                        "test_loss": test_loss,
                        "test_acc": test_acc
                    })

                    # release the memory

                    del model

    # print the results
    print(results)
    # save the results

    with open(f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_CNN_tuning.log", "w") as f:
        f.write(f"results: {results}\n")


    
    '''
    embedding_dim = 300
    # use LSTM model, defined in model.py
    glove = torchtext.vocab.GloVe(name='6B', dim=embedding_dim)
    # 获取词汇表大小
    vocab_size = len(vocab)
    pad_index = vocab["<pad>"]

    # 构建嵌入矩阵，大小为 [vocab_size, embedding_dim]
    embedding_matrix = torch.zeros(vocab_size, embedding_dim)

    # 为词汇表中的每个词填充对应的 GloVe 向量
    for i, token in enumerate(vocab.get_itos()):  # vocab.get_itos() 获取词汇表中的所有词（从索引到词的映射）
        if token in glove.stoi:  # 检查词是否在 GloVe 词汇表中
            embedding_matrix[i] = glove[token]
        else:
            embedding_matrix[i] = torch.randn(embedding_dim)  # 如果词不在 GloVe 中，用随机向量初始化
    embedding_matrix = embedding_matrix.float()
    # use LSTM model, defined in model.py
    model = LSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=256,
        output_dim=output_dim,
        n_layers=4,
        dropout_rate=0.3,
        pad_index=pad_index,
        pretrained_embeddings=embedding_matrix,
        freeze_embeddings=False  # 设置为 True 则不微调，设置为 False 则在训练中微调
    )

    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # You can adjust this hyperparameter

    criterion = nn.CrossEntropyLoss()

    metrics = collections.defaultdict(list)

    model = model.to(args.device)
    criterion = criterion.to(args.device)

    metrics_LSTM = collections.defaultdict(list)

    print("Now LSTM performance:\n")
    best_valid_acc_LSTM = 0
    for epoch in range(args.n_epochs):
        train_loss, train_acc = train(train_data_loader, model, criterion, optimizer, args.device)
        valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, args.device)
        metrics_LSTM["train_losses"].append(train_loss)
        metrics_LSTM["train_accs"].append(train_acc)
        metrics_LSTM["valid_losses"].append(valid_loss)
        metrics_LSTM["valid_accs"].append(valid_acc)

        if not metrics_LSTM["valid_accs"] or valid_acc > max(metrics_LSTM["valid_accs"]):
            torch.save(model.state_dict(), "best_model_LSTM.pth")
            best_valid_acc_LSTM = valid_acc

        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

        # save the model with the best valid acc
        if best_valid_acc_LSTM < valid_acc:
            torch.save(model.state_dict(), "best_model_LSTM.pth")
            best_valid_acc_LSTM = valid_acc
        # early stopping, if the val acc is not improved in 3 consecutive epochs, break the training process
    # testing

    test_loss, test_acc = evaluate(test_data_loader, model, criterion, args.device)
    print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")
    # clear the current figure
    plt.clf()
    # draw loss
    plt.plot(metrics_LSTM["train_losses"], label="train_loss", color="red")
    plt.plot(metrics_LSTM["valid_losses"], label="valid_loss", color="blue")
    # define the title
    plt.title("Loss")
    # define the x label
    plt.xlabel("Epoch")
    # define the y label
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # save the figure to the current directory
    plt.savefig("loss_LSTM.png")

    # clear the current figure
    plt.clf()

    # draw acc
    plt.plot(metrics_LSTM["train_accs"], label="train_acc", color="red")
    plt.plot(metrics_LSTM["valid_accs"], label="valid_acc", color="blue")
    # define the title
    plt.title("Accuracy")

    # define the x label
    plt.xlabel("Epoch")
    # define the y label
    plt.ylabel("Accuracy")

    plt.legend()
    plt.show()
    # save the figure to the current directory
    plt.savefig("acc_LSTM.png")

    # save the loss, acc, hyper-parameters into log with the timestamp as the filename
    with open(f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_LSTM.log", "w") as f:
        f.write(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}\n")
        f.write(f"hyper-parameters: {args}\n")
        f.write(f"metrics: {metrics_LSTM}\n")
        f.write(f"vocab: {vocab}\n")
        f.write(f"model: {model}\n")
        f.write(f"optimizer: {optimizer}\n")
        f.write(f"criterion: {criterion}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"special_tokens: {special_tokens}\n")
        f.write(f"unk_index: {unk_index}\n")
        f.write(f"pad_index: {pad_index}\n")
        f.write(f"vocab_size: {vocab_size}\n")
        f.write(f"output_dim: {output_dim}\n")

    # save the model
    torch.save(model.state_dict(), "model_LSTM.pth")

    # print the best valid acc
    # print(f"best_valid_acc_CNN: {best_valid_acc}")
    print(f"best_valid_acc_LSTM: {best_valid_acc_LSTM}")
    '''