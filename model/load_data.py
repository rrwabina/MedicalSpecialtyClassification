import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt 
import spacy
import warnings 
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

nlp = spacy.load('en_core_web_sm')


def GENERATE_DATALOADER(input_ids, attention_mask, labels, batch_size = 64, use_sampler = True):
    if use_sampler:
        oversampler = RandomOverSampler(random_state = 42)
        X = np.concatenate((input_ids, attention_mask), axis = -1)
        y = np.ravel(labels)

        X_resampled, y_resampled = oversampler.fit_resample(X, y)

        input_ids_resampled      = X_resampled[:, :input_ids.shape[1]]
        attention_mask_resampled = X_resampled[:, input_ids.shape[1]:]
        labels_resampled = y_resampled

        dataset = TensorDataset(torch.tensor(input_ids_resampled),
                                torch.tensor(attention_mask_resampled),
                                torch.tensor(labels_resampled))
        
        train_size = int(0.6 * len(dataset))
        valid_size = int(0.2 * len(dataset))
        tests_size = len(dataset) - train_size - valid_size
        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, tests_size])
        
    else:
        dataset = TensorDataset(torch.tensor(input_ids), 
                                torch.tensor(attention_mask), 
                                torch.tensor(labels))
        
        train_size = int(0.6 * len(dataset))
        valid_size = int(0.2 * len(dataset))
        tests_size = len(dataset) - train_size - valid_size

        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, tests_size])

    train_dataloader = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        batch_size = batch_size)
    validation_dataloader = DataLoader(
        valid_dataset,
        sampler = SequentialSampler(valid_dataset),
        batch_size = batch_size)
    test_dataloader = DataLoader(
        test_dataset,
        sampler = SequentialSampler(test_dataset),
        batch_size = batch_size)
    return train_dataloader, validation_dataloader, test_dataloader


class NLPDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        return sequence, label
    
def process_data(input_ids, labels):
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(input_ids, labels, test_size = 0.20, random_state = 42)
    train_sequences, val_sequences,  train_labels, val_labels  = train_test_split(train_sequences,  train_labels, test_size = 0.15, random_state = 42)

    BATCH_SIZE = 64

    train_dataset = NLPDataset(train_sequences, train_labels)
    valid_dataset = NLPDataset(val_sequences, val_labels)
    tests_dataset = NLPDataset(test_sequences, test_labels)

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle = False)
    tests_loader = DataLoader(tests_dataset, batch_size = BATCH_SIZE, shuffle = False)