
import transformers
from time import time
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
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

from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize


class GenerateNewTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.WordPiece(unk_token = '[UNK]'))
        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents()])
        
        self.tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        self.special_tokens = ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']
        self.trainer = trainers.WordPieceTrainer(
            vocab_size = 30000,
            special_tokens = self.special_tokens)
        
        self.cls_token_id = None
        self.sep_token_id = None

    def load_dataset(self, filepath):
        dataset = pd.read_csv(filepath)
        dataset['transcription'].fillna(dataset['description'], inplace = True)
        return dataset

    def get_training_corpus(self, dataset):
        for i in range(0, len(dataset), 1000):
            yield dataset[i: i + 1000]['transcription']

    def train_tokenizer(self, dataset):
        self.tokenizer.train_from_iterator(
            self.get_training_corpus(dataset),
            trainer = self.trainer)
        
        self.cls_token_id = self.tokenizer.token_to_id('[CLS]')
        self.sep_token_id = self.tokenizer.token_to_id('[SEP]')

        self.tokenizer.post_processor = processors.TemplateProcessing(
            single = f'[CLS]:0 $A:0 [SEP]:0',
            pair   = f'[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1',
            special_tokens = [('[CLS]', self.cls_token_id), ('[SEP]', self.sep_token_id)])
        self.tokenizer.decoder = decoders.WordPiece(prefix = '##')

    def save_tokenizer(self, filepath):
        self.tokenizer.save(filepath)

    def generate_tokenizer(self, dataset_filepath, tokenizer_filepath):
        dataset = self.load_dataset(dataset_filepath)
        self.train_tokenizer(dataset)
        self.save_tokenizer(tokenizer_filepath)

class CombinedTokenizer:
    def __init__(self, bert_tokenizer, medical_tokenizer):
        self.bert_tokenizer = bert_tokenizer
        self.medical_tokenizer = medical_tokenizer
    
    def tokenize(self, text):
        bert_tokens = self.bert_tokenizer.tokenize(text)
        medical_tokens = self.medical_tokenizer.tokenize(text)
        combined_tokens = bert_tokens + medical_tokens  
        return combined_tokens
