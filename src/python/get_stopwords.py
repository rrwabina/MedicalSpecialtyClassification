import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt 
import spacy
import re
import logging
import random
import os
import itertools
import warnings 
warnings.filterwarnings('ignore')

from spacy.lang.en.stop_words import STOP_WORDS
from spacy.pipeline.tagger import Tagger
from spacy.language import Language
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset
from imblearn.over_sampling import RandomOverSampler
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer 

import transformers
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from transformers import RobertaModel
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer)
from tokenizers import BertWordPieceTokenizer
from transformers import PreTrainedTokenizerFast
from transformers import BertTokenizerFast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

nlp = spacy.load('en_core_web_sm')

stopwords_path = '../data/clinical-stopwords.txt'
with open(stopwords_path, 'r', encoding = 'utf-8') as file:
    stopwords = [line.strip() for line in file]

@Language.component('remove_stopwords')
def remove_stopwords(doc):
    tokens = [token for token in doc if token.text.lower() not in stopwords]
    return spacy.tokens.Doc(doc.vocab, tokens)
nlp.add_pipe('remove_stopwords')

from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(data_iter):
    for text in data_iter:
        tokens = medical_tokenizer.tokenize(text)
        yield from tokens

text_column = pd.read_csv('../data/mtsamples.csv')['transcription'].astype('str')
vocab = build_vocab_from_iterator(yield_tokens(text_column), specials = ['<unk>', '<pad>', '<bos>', '<eos>'])
vocab.set_default_index(vocab['<unk>'])