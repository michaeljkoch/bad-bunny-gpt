import unicodedata
import re
from io import open


class Vocab:
    def __init__(self):
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

    @staticmethod
    def pad(seq, content, add_length):
        seq.extend([content] * (add_length - len(seq)))
        return seq

    @staticmethod
    def idx_from_sent(lang, sentence):
        return [lang.word_to_idx[word] for word in sentence.split(' ')]

    @staticmethod
    def tensors_from_lyrics(vocab, lyrics, max_seq_length):
        src = []
        trg = []
        for sentence in lyrics:
            src_indexes = Vocab.idx_from_sent(vocab, sentence)
            trg_indexes = Vocab.idx_from_sent(vocab, sentence)
            src_indexes.insert(0, vocab.word_to_idx['<sos>'])
            src_indexes = Vocab.pad(src_indexes, vocab.word_to_idx['<pad>'], max_seq_length)
            trg_indexes.append(vocab.word_to_idx['<eos>'])
            trg_indexes = Vocab.pad(trg_indexes, vocab.word_to_idx['<pad>'], max_seq_length)
            src.append(src_indexes)
            trg.append(trg_indexes)

        return src, trg

    @staticmethod
    def unicode_to_ascii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def normalize_string(s):
        s = s.lower().strip()
        s = re.sub(r"([.!?,)])", r" \1", s)
        s = re.sub(r"([¡¿(])", r"\1 ", s)
        # s = re.sub(r"[^a-zA-Z.!?,)¡¿(]+", r" ", s)
        return s

    @staticmethod
    def filter_lyric(lyric, max_len):
        return len(lyric.split(' ')) < max_len

    @staticmethod
    def filter_lyrics(lyrics, max_len):
        return [lyric for lyric in lyrics if Vocab.filter_lyric(lyric, max_len)]
