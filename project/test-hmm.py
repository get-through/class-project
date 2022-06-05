from re import M
from hamcrest import none
from utils_pos import get_word_tag, preprocess  
import pandas as pd
from collections import defaultdict
import math
import numpy as np
import sys
import string

# training dataset
with open("WSJ_02-21.pos", 'r') as f:
    training_corpus = f.readlines()

with open("nltk_generated_training_data.txt", 'r') as f:
    training_corpus += f.readlines()

# with open("WSJ_24.pos", 'r') as f:
#     training_corpus += f.readlines()

# with open("dict_training.txt", 'r') as f:
#     training_corpus += f.readlines()

# testing dataset
text = ''
while True:
    line_text = input()
    if line_text == '':
        break
    text += line_text + ' '
# print(text)
f = open("test_words.txt", 'w')
voc_l = text.split(' ')
# print(voc_l)
for i in range(len(voc_l)-1):
    if i:
        f.write('\n')
    for j in range(len(voc_l[i])):
        if voc_l[i][0] != '"' and (voc_l[i][0]<'A' or (voc_l[i][0]>'Z' and voc_l[i][0]<'a') or voc_l[i][0]>'z'):
            f.write(voc_l[i])
            break
        if (voc_l[i][j]>='a' and voc_l[i][j]<='z') or (voc_l[i][j]>='A' and voc_l[i][j]<='Z'):
            f.write(f"{voc_l[i][j]}")
print(text)
# sys.exit()

# vocab dictionary
with open("test_vocab.txt", 'r') as f:
    voc_l += f.read().split('\n')

with open("nltk_generated_vocabs.txt", 'r') as f:
    voc_l += f.read().split('\n')

# print(voc_l[0:10])
vocab={}
# sys.exit()

for i, word in enumerate(set(voc_l)): 
    vocab[word] = i

# print("Vocabulary dictionary, key is the word, value is a unique integer")
# cnt = 0
# for k,v in vocab.items():
#     print(f"{k}:{v}")
#     cnt += 1
#     if cnt > 50:
#         break

with open("WSJ_24.pos", 'r') as f:
    y = f.readlines()

_, prep = preprocess(vocab, "test_words.txt")


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

# def predict_pos(prep, y, emission_counts, vocab, states):
#     num_correct = 0
#     all_words = set(emission_counts.keys())
#     total = len(y)

#     for word, y_tup in zip(prep, y):
#         y_tup_l = y_tup.split()

#         if len(y_tup_l) == 2:
#             true_label = y_tup_l[1]
#         else:
#             continue

#         count_final = 0
#         pos_final = ''
    
#         if word in vocab:
#             for pos in states:

#                 key = (pos, word)

#                 if key in emission_counts:
#                     count = emission_counts[key]

#                     if count > count_final:
#                         count_final = count
#                         pos_final = pos
#             # print(word, pos_final)
#             if true_label == pos_final:
#                 num_correct += 1

#     accuracy = num_correct / total

#     return accuracy

# accuracy_predict_pos = predict_pos(prep, y, emission_counts, vocab, states)
# print(f"Accuracy of prediction using predict_pos is {accuracy_predict_pos:.4f}")

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

print('start processing')
pred = viterbi(states, tag_counts, A, B, prep, vocab)
print(pred)
print('22')

# def compute_accuracy(pred, y):
#     num_correct = 0
#     total = 0
#     for prediction, y in zip(pred, y):
#         word_tag_tuple = y.split()

#         if len(word_tag_tuple) < 2:
#             continue

#         tag = word_tag_tuple[1]

#         if tag == prediction:
#             num_correct += 1
        
#         total += 1
    
#     return num_correct / total

# print(f"Accuracy of the Viterbi algorithm is {compute_accuracy(pred, y):.4f}")