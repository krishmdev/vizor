# Vendored from https://github.com/GEO-optim/GEO (src/utils.py, commit c9e985f2bc4b539a01e8e9d226ff2a3d8d29a888).
# Copyright the GEO authors (Aggarwal, Murahari, Rajpurohit, Kalyan, Narasimhan, Deshpande).
# Licensed under the Apache License, Version 2.0; see NOTICE at the repository root and
# http://www.apache.org/licenses/LICENSE-2.0. Only the impression_*_simple functions and
# get_num_words are copied, unmodified, so the test suite can check parity against them.
import itertools
import math


def get_num_words(line):
    return len([x for x in line if len(x)>2])

def impression_wordpos_count_simple(sentences, n = 5, normalize=True):
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = get_num_words(sent[0])
            score *= math.exp(-1 * i / (len(sentences)-1)) if len(sentences)>1 else 1
            score /= len(sent[2])

            try: scores[cit-1] += score
            except: print(f'Citation Hallucinated: {cit}')
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores

def impression_word_count_simple(sentences, n = 5, normalize=True):
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = get_num_words(sent[0])
            score /= len(sent[2])
            try: scores[cit-1] += score
            except: print(f'Citation Hallucinated: {cit}')
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores
    

def impression_pos_count_simple(sentences, n = 5, normalize=True):
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = 1
            score *= math.exp(-1 * i / (len(sentences)-1)) if len(sentences)>1 else 1
            score /= len(sent[2])
            try: scores[cit-1] += score
            except: print(f'Citation Hallucinated: {cit}')
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores
                

