# Defined model class
import json

with open('./data/config.json', 'r') as f:
    config_data = json.load(f)

# Extract variables
classes = config_data['classes']
max_len = config_data['max_len']
vocab_size = config_data['vocab_size']

num_lstm = 1
num_hidden = 32
embedding_size = 128

import torch.nn as nn

class SentimentAnalysisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(embedding_size, num_hidden, bidirectional = True, num_layers=num_lstm)
        self.linear = nn.Linear(2 * num_hidden * max_len, len(classes))
        self.dropout = nn.Dropout(p=0.35)

    def forward(self, x):
        logits = self.embd(x)
        logits , (h_n, c_n) = self.lstm(logits)
        logits = logits.flatten(start_dim = 1, end_dim=-1)
        logits = self.linear(logits)
        
        softmax = nn.Softmax(dim=1)
        probability = softmax(logits)
        
        return logits, probability