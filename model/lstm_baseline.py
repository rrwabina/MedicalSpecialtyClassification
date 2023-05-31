import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import transformers
from time import time
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
from transformers import RobertaModel, BertModel
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

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        _, (hidden, _) = self.lstm(inputs)
        hidden = hidden.squeeze(0)  
        output = self.fc(hidden)
        return output
    
class LSTMClassifierModified(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(LSTMClassifierModified, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 32)
        self.fc3 = torch.nn.Linear(32, out_features)
                
    def forward(self, inputs):
        x = F.relu(self.fc1(inputs.squeeze(1)))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = F.relu(logits)
        return probs
    
class LSTMClassifierExtended(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifierExtended, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        _, (hidden, _) = self.lstm(inputs)
        hidden = hidden[-1] 
        output = self.fc(hidden)
        return output

def BERT_EMBEDDING(input_ids, attention_mask, token_type_ids):
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(input_ids = input_ids, 
                             attention_mask = attention_mask, 
                             token_type_ids = token_type_ids)
        bert_embeddings = outputs.last_hidden_state

    batch_size = bert_embeddings.size(0)
    sequence_length = bert_embeddings.size(1)
    bert_embeddings = bert_embeddings.view(batch_size, sequence_length, -1)
    embeddings  = bert_embeddings.permute(1, 0, 2)
    return bert_model, embeddings

def ROBERTA_EMBEDDING(input_ids, attention_mask, token_type_ids):
    model_name = 'roberta-base'
    model = RobertaModel.from_pretrained(model_name)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids = input_ids,
                        attention_mask = attention_mask,
                        token_type_ids = token_type_ids)
        embeddings = outputs.last_hidden_state

    batch_size = embeddings.size(0)
    sequence_length = embeddings.size(1)
    embeddings = embeddings.view(batch_size, sequence_length, -1)
    embeddings = embeddings.permute(1, 0, 2)
    return model, embeddings

def LSTM_BASELINES(bert_model, embeddings, basic_classifier = False):
    input_size = bert_model.config.hidden_size
    hidden_size, num_classes = 50, 9
    if basic_classifier:
        lstm_model   = LSTMClassifierModified(input_size, hidden_size, num_classes)
    else:
        lstm_model   = LSTMClassifier(input_size, hidden_size, num_classes)
    lstm_output  = lstm_model(embeddings)
    output_probs = nn.functional.softmax(lstm_output, dim = 0)
    _, predicted_labels = torch.max(output_probs, dim = 0)
    return output_probs, predicted_labels