from website import create_app

import pandas as pd
raw_data = pd.read_csv("./data/sentiments.csv")

df = raw_data.dropna()
df.sample(frac = 1).head()
print(len(df))

classes = df['status'].unique()
classes

# Another pre-requisites
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(oov_token='UNK', lower = True)
tokenizer.fit_on_texts(df['statement'].values)

max_len = max([len(x) for x in tokenizer.texts_to_sequences(df['statement'].values)])   # 5421

vocab_size = len(tokenizer.word_index) + 1

# Defined model class
import torch.nn as nn
import torch.nn.functional as F

num_lstm = 1
num_hidden = 64
embedding_size = 64

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(embedding_size, num_hidden, bidirectional = True, num_layers=num_lstm)
        self.linear = nn.Linear(2 * num_hidden * max_len, len(classes))

    def forward(self, x):
        x = self.embd(x)
        x , (h_n, c_n) = self.lstm(x)
        x = x.flatten(start_dim = 1, end_dim=-1)
        x = self.linear(x)
        return x

if __name__ == "__main__" :
    app = create_app()
    # app.run(debug=True)   # auto reload jk file berubah, tdk bs utk image recognition
    app.run(debug=False)