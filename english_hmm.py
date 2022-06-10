from re import M
from hamcrest import none
from utils_pos import get_word_tag, preprocess  
import pandas as pd
from collections import defaultdict
import math
import numpy as np
import sys
import string
import nltk
from nltk import pos_tag, word_tokenize

# training dataset
with open("WSJ_02-21.pos", 'r') as f:
    training_corpus = f.readlines()

with open("nltk_generated_training_data.txt", 'r') as f:
    training_corpus += f.readlines()

with open("nltk_generated_vocabs.txt", 'r') as f:
    voc_l = f.read().split('\n')

with open("hmm_vocab.txt", 'r') as f:
    voc_l += f.read().split('\n')

vocab={}

for i, word in enumerate(sorted(set(voc_l))): 
    vocab[word] = i

# input
text = ''
while True:
    line_text = input()
    if line_text == '':
        break
    text += line_text + ' '
f = open("test_words.txt", 'w')
words = text.split(' ')
for i in range(len(words)-1):
    if i:
        f.write('\n')
    prev_not_alphabet=0
    for j in range(len(words[i])):
        if (words[i][j]<'A' or (words[i][j]>'Z' and words[i][j]<'a') or words[i][j]>'z'):
            if prev_not_alphabet==0:
                f.write('\n')
                # print("1")    
                prev_not_alphabet=1
        elif prev_not_alphabet==1 and words[i][j-1]!='\'':
            f.write('\n')
            prev_not_alphabet=0
        f.write(f"{words[i][j]}")
        if words[i][j]=='.' or words[i][j]=='!' or words[i][j]=='?':
            f.write('\n')
print(text)
f.close()

tokens = nltk.word_tokenize(text)
listling = nltk.pos_tag(tokens)
_, prep = preprocess(vocab, "test_words.txt")
# print(f"prep={prep[0:20]}")

# training data processed into two matrixes
# and a tag_counts for recording all the status types
def create_dictionaries(training_corpus, vocab):
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    prev_tag = '--s--'
    i = 0

    for word_tag in training_corpus:
        i += 1
        word, tag = get_word_tag(word_tag, vocab)
        # print(word, tag)
        transition_counts[(prev_tag, tag)] += 1
        emission_counts[(tag, word)] += 1
        tag_counts[tag] += 1

        prev_tag = tag

    return emission_counts, transition_counts, tag_counts

emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)
# print(transition_counts)

states = sorted(tag_counts.keys())

#matrix A
def create_transition_matrix(alpha, tag_counts, transition_counts):
    all_tags = sorted(tag_counts.keys())
    num_tags = len(all_tags)
    A = np.zeros((num_tags, num_tags))

    trans_keys = set(transition_counts.keys())

    for i in range(num_tags):
        for j in range(num_tags):
            count = 0
            key = (all_tags[i], all_tags[j])
            if key in transition_counts:
                count = transition_counts[key]
            count_prev_tag = tag_counts[all_tags[i]]
            A[i,j] = (count+alpha) / (count_prev_tag+alpha*num_tags)
    return A

alpha = 0.001
A = create_transition_matrix(alpha, tag_counts, transition_counts)
# print(f"A:{A}")

# matrix B
def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    num_tags = len(tag_counts)
    all_tags = sorted(tag_counts.keys())
    num_words = len(vocab)
    B = np.zeros((num_tags, num_words))

    emis_keys = set(list(emission_counts.keys()))
    for i in range(num_tags):
        for j in range(num_words):
            count = 0
            key = (all_tags[i], vocab[j])
            if key in emis_keys:
                count = emission_counts[key]

            count_tag = tag_counts[all_tags[i]]

            B[i,j] = (count+alpha) / (count_tag+alpha*num_words)

    return B

B = create_emission_matrix(alpha, tag_counts, emission_counts, list(vocab))
# print(f"B:{B}")

# Viterbi
def viterbi(states, tag_counts, A, B, corpus, vocab):
    # initialize
    num_tags = len(tag_counts)
    best_probs = np.zeros((num_tags, len(corpus)))
    best_paths = np.zeros((num_tags, len(corpus)), dtype=int)
    s_idx = states.index("--s--")

    for i in range(num_tags):
        if A[s_idx,i]==0:
            best_probs[i,0] = float('-inf')
        else:
            best_probs[i,0] = math.log(A[s_idx,i]) + math.log(B[i,vocab[corpus[0]]])
    
    # forward
    for i in range(1, len(corpus)):
        if i % 5000 == 0:
            print("Words processed: {:>8}".format(i))
        for j in range(num_tags):
            best_prob_i = float('-inf')
            best_path_i = none

            for k in range(num_tags):
                prob = best_probs[k,i-1] + math.log(A[k,j]) + math.log(B[j,vocab[corpus[i]]])
            
                if prob > best_prob_i:
                    best_prob_i = prob
                    best_path_i = k
            
            best_probs[j,i] = best_prob_i
            best_paths[j,i] = best_path_i
    
    # backward
    m = len(corpus)
    z = [None] * m
    pred = [None] * m
    best_prob_of_last_word = float('-inf')

    for k in range(num_tags):
        if best_probs[k,m-1]>best_prob_of_last_word:
            best_prob_of_last_word = best_probs[k,m-1]
            z[m-1] = k

    pred[m-1] = states[z[m-1]]
    for i in range(m-1, 0, -1):
        tag_for_wordi = best_paths[z[i],i] 
        z[i-1] = tag_for_wordi
        pred[i-1] = states[z[i-1]]
    return pred

print('start tagging')
pred = viterbi(states, tag_counts, A, B, prep, vocab)
print(pred)
print(f"nltk result:")
for i, j in listling:
    print(i,'/',j,end='\t')