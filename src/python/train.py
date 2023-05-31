import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt 
import re
import logging
import random
import os
import itertools
import copy
import time
import warnings 
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datasets import load_dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertTokenizer, DistilBertModel
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer 

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
    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_accuracy(preds, y):
    batch_corr = (preds == y).sum()
    acc = batch_corr / len(y)
    return acc

def eval_predictions(predictions, labels):
    predicted_labels = torch.argmax(predictions, dim = 1)
    true_labels = labels.numpy()

    accuracy  = accuracy_score(true_labels,  predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average = 'weighted')
    recall = recall_score(true_labels, predicted_labels, average = 'weighted')
    f1 = f1_score(true_labels, predicted_labels, average = 'weighted')
    return {
        'Accuracy':     np.round(accuracy, 4),
        'Precision':    np.round(precision, 4),
        'Recall':       np.round(recall, 4),
        'F1-score':     np.round(f1, 4)}

def plot_metrics(train_losses, valid_losses, train_accurs, valid_accurs):
    alpha = 0.3
    smoothed_train_losses = [train_losses[0]]
    smoothed_valid_losses = [valid_losses[0]]
    smoothed_train_accurs = [train_accurs[0]]
    smoothed_valid_accurs = [valid_accurs[0]]
    
    for i in range(1, len(train_losses)):
        smoothed_train_losses.append(alpha * train_losses[i] + (1-alpha) * smoothed_train_losses[-1])
        smoothed_valid_losses.append(alpha * valid_losses[i] + (1-alpha) * smoothed_valid_losses[-1])
        smoothed_train_accurs.append(alpha * train_accurs[i] + (1-alpha) * smoothed_train_accurs[-1])
        smoothed_valid_accurs.append(alpha * valid_accurs[i] + (1-alpha) * smoothed_valid_accurs[-1])
    
    smoothed_train_losses = train_losses
    smoothed_train_accurs = train_accurs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))
    ax1.plot(smoothed_train_losses, label = 'Train')
    ax1.plot(smoothed_valid_losses, label = 'Valid')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Losses')
    ax1.legend()

    ax2.plot(smoothed_train_accurs, label='Train')
    ax2.plot(smoothed_valid_accurs, label='Valid')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracies')
    ax2.legend()
    plt.show()

def _train(model, loader, optimizer, criterion, batch_size = 16, device = 'cpu'):
    epoch_train_loss = 0
    epoch_train_accu = 0
    model.train()
    epoch_train_prediction = []

    for idx, data in enumerate(loader):
        inputs, attens, labels = data
        inputs, attens, labels = inputs.to(device), attens.to(device), labels.to(device, dtype = torch.long)
        optimizer.zero_grad()

        outputs = model(input_ids = inputs, attention_mask = attens)
        embedds = outputs.last_hidden_state

        batch_size, seq_length = embedds.size(0), embedds.size(1)
        embeddings = embedds.view(batch_size, seq_length, -1)
        embeddings = embeddings.permute(1, 0, 2)

        input_size = model.config.hidden_size
        hidden_size, num_classes = 50, 9

        lstm_model = LSTMClassifier(input_size, hidden_size, num_classes)
        lstm_output  = lstm_model(embeddings)
        loss = criterion(lstm_output, labels)

        output_probs = nn.functional.softmax(lstm_output, dim = 0)    
        _, predicted_labels = torch.max(output_probs, dim = 1) 

        loss.backward()
        optimizer.step()
          
        epoch_train_prediction.append(predicted_labels)
        accuracy = get_accuracy(predicted_labels, labels) 
        loss = np.round(loss.item(), 3)
        epoch_train_loss += loss.item()
        epoch_train_accu += accuracy.item()
    epoch_train_loss = epoch_train_loss / len(loader)
    epoch_train_accu = epoch_train_accu / len(loader)
    return epoch_train_loss, epoch_train_accu, epoch_train_prediction
    
def _evals(model, loader, criterion, batch_size = 64, device = 'cpu', display = False):
    epoch_valid_loss = 0
    epoch_valid_accu = 0
    model.eval()
    epoch_valid_prediction = []
    with torch.no_grad():
        for idx, data in enumerate(loader):
            inputs, attens, labels = data 
            inputs, attens, labels = inputs.to(device), attens.to(device), labels.to(device,  dtype = torch.long)

            outputs = model(input_ids = inputs, attention_mask = attens)
            embedds = outputs.last_hidden_state

            batch_size, seq_length = embedds.size(0), embedds.size(1)
            embeddings = embedds.view(batch_size, seq_length, -1)
            embeddings = embeddings.permute(1, 0, 2)  

            
            input_size = model.config.hidden_size
            hidden_size, num_classes = 128, 9
            lstm_model = LSTMClassifier(input_size, hidden_size, num_classes)
            lstm_output  = lstm_model(embeddings)
            loss = criterion(lstm_output, labels)
            loss = np.round(loss.item(), 3)

            output_probs = nn.functional.softmax(lstm_output, dim = 0)    
            _, predicted_labels = torch.max(output_probs, dim = 1)   
            epoch_valid_prediction.append(predicted_labels)
            accuracy = np.round(get_accuracy(predicted_labels, labels), 5)
            epoch_valid_loss += loss.item()
            epoch_valid_accu += accuracy.item()
    epoch_valid_loss = epoch_valid_loss / len(loader)
    epoch_valid_accu = epoch_valid_accu / len(loader)
    if display:
        print(f'Loss: {loss} \t Accuracy: {accuracy}')
    return epoch_valid_loss, epoch_valid_accu, epoch_valid_prediction

def train(num_epochs, model, train_loader, valid_loader, test_loader, optimizer, criterion, device, accuracy = True):
    best_valid_loss = float('inf')
    train_losses, valid_losses = [], []
    train_accurs, valid_accurs = [], []
    trainpredict, testspredict = [], []

    epoch_times = []
    list_best_epochs = []
    start = time()

    for epoch in range(num_epochs):

        train_loss, train_accu, tr_predict = _train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_accu, ts_predict = _evals(model, valid_loader, criterion, device)
        
        if accuracy:
            print(f'Epoch: {epoch + 1} \t Training: Loss {np.round(train_loss, 5)}   \t Accuracy: {np.round(train_accu, 5)} \t Validation Loss  {np.round(valid_loss, 5)} \t Accuracy: {np.round(valid_accu, 5)}')
        else:
            print(f'Epoch: {epoch + 1} \t Training: Loss {np.round(train_loss, 5)} \t Validation Loss  {np.round(valid_loss, 5)}')
        
        train_losses.append(train_loss)
        train_accurs.append(train_accu)
        valid_losses.append(valid_loss)
        valid_accurs.append(valid_accu)
        trainpredict.append(tr_predict)
        testspredict.append(ts_predict)

        end_time = time()
        epoch_mins, epoch_secs = epoch_time(start, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        list_best_epochs.append(best_epoch)
    test_loss, test_accu, test_predict  = _evals(best_model, test_loader, criterion, device)
    print(f'Training time: {np.round(time() - start, 4)} seconds')
    print(f'Final Best Model from Best Epoch {best_epoch + 1} Test Loss = {test_loss}, Test Accuracy = {test_accu}')
    return train_losses, valid_losses, train_accurs, valid_accurs, test_loss, test_accu, best_epoch, epoch_times, test_predict, best_model