# import string
import  numpy as np
import pandas as pd
import nltk
import sys
import jieba

# with open("11.txt", 'r') as f:
#     y = f.read().split('\n')

# with open("11.txt", 'r') as f:
#     y = y + f.read().split('\n')

# i = 0

# with open("file.txt", 'w') as ff:

#     while i < len(y):
#         ff.write(y[i].split('\t')[0]+"\n")
#         i += 1

# a=np.arange(10).reshape(2,5)
# # print(a)
# a[np.ix_([0,1],[1,3])]=3
# # print(a)

# # [[0 1 2 3 4]
# #  [5 6 7 8 9]]

# # [[0 3 2 3 4]
# #  [5 3 7 3 9]]

# arr = np.arange(32).reshape((8,4))
# print(arr)
# print(arr[np.ix_([1,5,7,2],[0,3,1,2])])

# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]
#  [16 17 18 19]
#  [20 21 22 23]
#  [24 25 26 27]
#  [28 29 30 31]]

# [[ 4  7  5  6]
#  [20 23 21 22]
#  [28 31 29 30]
#  [ 8 11  9 10]]

# matrix = [(1,2,3),(4,5,6),(7,8,9)]
# aa = pd.DataFrame(data=matrix, index=[1,3,2], columns=range(3))
# print(aa)
# #    0  1  2
# # 1  1  2  3
# # 3  4  5  6
# # 2  7  8  9


# data collecting
# f = open("passage_temp.txt",'r')
# # x = open("nltk_generated_vocabs.txt",'w')
# words = f.readlines()
# text = ""
# for i in range(len(words)):
#     text += words[i]
#     # ee = words[i].split()
#     # x.write(f"{ee[0]}\n")
# # print(text)

# tokens = nltk.word_tokenize(text)
# y = open("nltk_generated_training_data.txt",'a')
# z = open("nltk_generated_vocabs.txt",'a')
# listling = nltk.pos_tag(tokens)
# for i in range(len(listling)):
#     y.write(f"{listling[i][0]}\t{listling[i][1]}\n")
#     z.write(f"{listling[i][0]}\n")


# with open("tag_help.txt", 'w') as f:
#     sys.stdout = f
#     nltk.help.upenn_tagset()

# import nltk
# from nltk import pos_tag, word_tokenize
# from nltk.corpus import brown

# training_corpus = nltk.corpus.brown
# print(training_corpus)

sentence="比如最常用的一个示例"
data1=jieba.lcut(sentence, cut_all=False)
print(data1)