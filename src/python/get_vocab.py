import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt 
import spacy
import warnings 
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize



SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

nlp = spacy.load('en_core_web_sm')

def initialize_custom_vocabs(path = '../data/SNMI.csv'):
    clinical = pd.read_csv(path)
    features = ['Preferred Label','Synonyms']
    clinical = clinical[features]
    clinical = clinical['Preferred Label'].append(clinical['Synonyms'])
    clinical = clinical.dropna()
    
    vocab = clinical.str.split('\W', expand = True).stack().unique()
    vocab = filter(None, vocab)

    filepath = '../data/vocab.txt'
    with open(filepath, 'w') as file_handler:
        for item in vocab:
            file_handler.write('{}\n'.format(item))

    with open('../data/vocab.txt', 'r') as f:
        vocab_words = set(f.read().splitlines())
    return vocab_words  

vocab_words = initialize_custom_vocabs()
def is_vocab_word(token):
    return token.lower_ in vocab_words

execute = True
if execute:
    spacy.tokens.Token.set_extension('is_vocab', getter = is_vocab_word, force = True)

def medical_vocabs(text):
    doc = nlp(text)
    tagged_tokens = [(token.text, token.pos_) for token in doc]
    filtered_tokens = [(token, pos) for token, pos in tagged_tokens if token._.is_vocab]
    return filtered_tokens

def generate_index2word(vocab_words):
    word2index = {'<PAD>': 0, 
                  '<UNK>': 1}
    for vo in vocab_words:
        if word2index.get(vo) is None:
            word2index[vo] = len(word2index)
            
    index2word = {v:k for k, v in word2index.items()}
    return index2word

def initialize_custom_tagger(path = '../data/clinical-stopwords.txt'):
    with open(path, 'r') as f:
        stop_words = set(f.read().splitlines())
    return stop_words

stop_words = initialize_custom_tagger()
def is_stop_word(token):
    return token.lower_ in stop_words

execute = True
if execute:
    spacy.tokens.Token.set_extension('is_stop', getter = is_stop_word, force = True)

def medical_tagger(text):
    doc = nlp(text)
    for token in doc:
        if token.lower_ in stop_words:
            token.is_stop = True
        else:
            token.is_stop = False
    return doc

for word in vocab_words:
    nlp.vocab[word]
vocab = nlp.vocab

stop_tagger  = Tagger(nlp.vocab, medical_tagger)
vocab_tagger = Tagger(nlp.vocab, medical_vocabs)
excluded_tokens = {}

use_stop_tagger, use_vocab_tagger  = False, False
if use_stop_tagger:
    nlp.add_pipe('stop_tagger', config = {'component': stop_tagger}, last = True)
    excluded_tokens.add('is_stop')

if use_vocab_tagger:
    nlp.add_pipe(name = 'vocab tagger',
                 component = vocab_tagger,
                 remote = True
                )
    excluded_tokens['is_vocab'] = {False}