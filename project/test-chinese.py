from re import M
from hamcrest import none
from utils_pos import get_word_tag, preprocess  
import pandas as pd
from collections import defaultdict
import math
import numpy as np
import string

with open("pku_training.utf8", 'r', encoding='utf-8') as f:
    text = f.read().split('\n')

y = open("parsed_vocab.utf8", 'w', encoding='utf-8')
x = open("chinese_vocab.utf8", 'w', encoding='utf-8')

words = []

for i in range(len(text)):
    words += text[i].split("  ")

for i in range(len(words)):
    y.write(f"{words[i]}\n")

words = list(set(words))

for i in range(len(words)):
    x.write(f"{words[i]}\n")
