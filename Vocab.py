import unicodedata
import string
import re
from io import open
import torch


class Vocab:
    def __init__(self, name):
        self.name = name
        self.word_to_idx = dict()
        self.word_to_idx['<pad>'] = 0
        self.word_to_idx['<unk>'] = 1
        self.word_to_idx['<sos>'] = 2
        self.word_to_idx['<eos>'] = 3
        self.n_words = 4
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.word_to_count = dict()

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.n_words
            self.word_to_count[word] = 1
            self.idx_to_word[self.n_words] = word
            self.n_words += 1
        else:
            self.word_to_count[word] += 1


def pad(seq, content, add_length):
    seq.extend([content] * (add_length - len(seq)))
    return seq


def idx_from_sent(lang, sentence):
    return [lang.word_to_idx[word] for word in sentence.split(' ')]


def tensor_from_sent(lang, sentence, max_seq_length, is_train, is_src):
    indexes = idx_from_sent(lang, sentence)
    if is_train:
        if is_src:
            indexes.insert(0, lang.word_to_idx['<sos>'])
            indexes = pad(indexes, lang.word_to_idx['<pad>'], max_seq_length)
        else:
            indexes.append(lang.word_to_idx['<eos>'])
            indexes = pad(indexes, lang.word_to_idx['<pad>'], max_seq_length)
    else:
        indexes.insert(0, lang.word_to_idx['<sos>'])
    return indexes


def tensors_from_lyrics(lang, lyrics, max_seq_length, is_train, is_src):
    src_tensors = [tensor_from_sent(lang, sentence, max_seq_length, is_train, is_src) for sentence in lyrics]
    return src_tensors


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?,)])", r" \1", s)
    s = re.sub(r"([¡¿(])", r"\1 ", s)
    # s = re.sub(r"[^a-zA-Z.!?,)¡¿(]+", r" ", s)
    return s


def filter_lyric(lyric, max_len):
    return len(lyric.split(' ')) < max_len


def filter_lyrics(lyrics, max_len):
    return [lyric for lyric in lyrics if filter_lyric(lyric, max_len)]


def read_lyrics(lyrics):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(f'/Users/michaelkoch/Documents/Academia/Projects/Transformer-Implementation/data/{lyrics}.txt',
                 encoding='utf-8').read().strip().split('\n')

    # print(lines)

    sentences = [normalize_string(line) for line in lines]

    processed_lyrics = filter_lyrics(sentences, 20)

    src = Vocab(lyrics)

    for lyric in processed_lyrics:
        src.add_sentence(lyric)

    print("Counted words:")
    print(src.n_words)

    return src, processed_lyrics


if __name__ == "__main__":
    read_lyrics('lyrics-final')
