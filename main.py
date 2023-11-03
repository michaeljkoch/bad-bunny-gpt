#!/usr/bin/python3

"""
Created on: 03 Nov 2023 14:22:04
By: Michael Koch
"""

import Vocab

if __name__ == "__main__":
    v, lyrics = Vocab.read_lyrics()

    src, trg = Vocab.tensors_from_lyrics(v, lyrics, 20)

    print(src[1000])
    print(trg[1000])
