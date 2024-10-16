from website import create_app

# Defined model class
import torch.nn as nn

# num_lstm = 1
# num_hidden = 64
# embedding_size = 64

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.embd = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        # self.lstm = nn.LSTM(embedding_size, num_hidden, bidirectional = True, num_layers=num_lstm)
        # self.linear = nn.Linear(2 * num_hidden * max_len, len(classes))

    def forward(self, x):
        x = self.embd(x)
        x , (h_n, c_n) = self.lstm(x)
        x = x.flatten(start_dim = 1, end_dim=-1)
        x = self.linear(x)
        return x

if __name__ == "__main__" :
    app = create_app()
    # app.run(debug=True)   # auto reload jk file berubah, tdk bs krn auto ada py cache
    app.run(debug=False)