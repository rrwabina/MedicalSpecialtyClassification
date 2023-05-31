import torch
import numpy as np
from transformers import BertTokenizer
from datasets import load_dataset

def preprocess(example):
    max_sequence_length = 256
    tokenized_transcription = tokenizer(example['transcription'], 
                                        truncation = True, 
                                        max_length = max_sequence_length, 
                                        padding = 'max_length')
    return {'input_ids': tokenized_transcription['input_ids'],
            'attention_mask': tokenized_transcription['attention_mask'],
            'medical_specialty': example['medical_specialty']}

def tokenize_and_split(examples):
    return tokenizer(
        examples['transcription'],
        truncation = True,
        max_length = 256,
        return_overflowing_tokens = True)

dataset = load_dataset('csv', data_files = '../data/mtsamples_modified.csv')
dataset_streamed = load_dataset('csv', data_files = '../data/mtsamples_modified.csv', streaming = True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenized_dataset = dataset_streamed.map(preprocess, batched = True, batch_size = 16)
tokenized_dataset = tokenized_dataset.shuffle(buffer_size = 10_000, seed = 42)

for split in dataset.keys():
    assert len(dataset[split]) == len(dataset[split].unique('Unnamed: 0'))
dataset = dataset.rename_column(original_column_name = 'Unnamed: 0', 
                                new_column_name = 'patient_id')
tokenized_dataset = dataset.map(tokenize_and_split, 
                                batched = True, 
                                remove_columns = dataset['train'].column_names)

input_ids = np.array(tokenized_dataset['train']['input_ids'])
sequence_length = max(len(ids) for ids in input_ids)
input_ids = [ids + [0] * (sequence_length - len(ids)) for ids in input_ids]
input_ids = torch.tensor(input_ids)

attention_mask = tokenized_dataset['train']['attention_mask']
attention_mask = [mask + [0] * (sequence_length - len(mask)) for mask in attention_mask]
attention_mask = torch.tensor(attention_mask)

token_type_ids = tokenized_dataset['train']['token_type_ids']
token_type_ids = [mask + [0] * (sequence_length - len(mask)) for mask in token_type_ids]
token_type_ids = torch.tensor(token_type_ids)