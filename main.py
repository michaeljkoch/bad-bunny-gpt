#!/usr/bin/python3

"""
Created on: 03 Nov 2023 14:22:04
By: Michael Koch
"""

import Vocab


def read_lyrics():
    print("Reading lines...")

    lines = open('data/lyrics.txt', encoding='utf-8').read().strip().split('\n')

    vocab = Vocab.Vocab()
    sentences = [vocab.normalize_string(line) for line in lines]
    processed_lyrics = vocab.filter_lyrics(sentences, 20)

    for lyric in processed_lyrics:
        vocab.add_sentence(lyric)

    print("Counted words:")
    print(vocab.n_words)

    return vocab, processed_lyrics


if __name__ == "__main__":
    vocab, lyrics = read_lyrics()

    src, trg = vocab.tensors_from_lyrics(vocab, lyrics, 20)

    # check that target sent is shifted right for training
    print(src[1000])
    print(trg[1000])
