#!/usr/bin/python3

"""
Created on: 03 Nov 2023 14:22:04
By: Michael Koch
"""

import torch
import torch.nn as nn
import Vocab
from Transformer import Transformer

import random
import time
import math


def read_lyrics():
    print("Reading lines...")

    lines = open('data/lyrics.txt', encoding='utf-8').read().strip().split('\n')

    vocab = Vocab.Vocab()
    sentences = [vocab.normalize_string(line) for line in lines]
    processed_lyrics = vocab.filter_lyrics(sentences, 20)

    for lyric in processed_lyrics:
        vocab.add_sentence(lyric)

    print("Counted words: ", vocab.n_words)

    return vocab, processed_lyrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(voacb, model, src_train, trg_train, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for i, src in enumerate(src_train):
        src = torch.tensor(src)

        optimizer.zero_grad()

        preds = model(src.unsqueeze(0))
        # print("preds:")
        # print("\t", preds.size())

        output_dim = preds.shape[-1]

        preds = preds.contiguous().view(-1, output_dim)
        pred_idx = preds.argmax(1).tolist()
        print(pred_idx)
        pred_words = [vocab.idx_to_word[idx] for idx in pred_idx]
        print(pred_words)
        trg = torch.tensor(trg_train[i])

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(preds, trg)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(src_train)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    vocab, lyrics = read_lyrics()

    src, trg = vocab.tensors_from_lyrics(lyrics, 20)

    # print(round(len(src)*0.9))
    train_idx = random.sample(range(0, len(src)), round(len(src)*0.9))
    print(train_idx)
    # print(train_idx)
    # src_train = torch.tensor([src[i] for i in train_idx])
    # trg_train = torch.tensor([trg[i] for i in train_idx])
    # src_test = torch.tensor([src[i] for i in range(len(src)) if i not in train_idx])
    # trg_test = torch.tensor([trg[i] for i in range(len(src)) if i not in train_idx])

    src_train = [src[i] for i in train_idx]
    trg_train = [trg[i] for i in train_idx]
    src_test = [src[i] for i in range(len(src)) if i not in train_idx]
    trg_test = [trg[i] for i in range(len(src)) if i not in train_idx]

    # check that target sent is shifted right for training
    # print(src[480])
    # print(trg[480])

    vocab_size = vocab.n_words
    model_dim = 128
    num_heads = 4
    num_layers = 3
    ff_dim = 64
    max_seq_len = 20
    LEARNING_RATE = 0.0005
    N_EPOCHS = 10
    BATCH_SIZE = 8

    model = Transformer(vocab_size, model_dim, num_heads, num_layers, ff_dim, max_seq_len, dr=0.9)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train(vocab, model, src_train, trg_train, optimizer, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
