from website import create_app

# Defined model class
import torch.nn as nn

class SentimentAnalysisModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        logits = self.embd(x)
        logits , (h_n, c_n) = self.lstm(logits)
        logits = logits.flatten(start_dim = 1, end_dim=-1)
        logits = self.linear(logits)
        
        softmax = nn.Softmax(dim=1)
        probability = softmax(logits)
        
        return logits, probability

if __name__ == "__main__" :
    app = create_app()
    # app.run(debug=True)   # auto reload jk file berubah, tdk bs krn auto ada pycache
    app.run(debug=False)