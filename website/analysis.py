from flask import Blueprint, render_template, request, redirect

analysis = Blueprint("analysis", __name__)

import pandas as pd
raw_data = pd.read_csv("./data/sentiments.csv")

df = raw_data.dropna()
df.sample(frac = 1).head()
print(len(df))

classes = df['status'].unique()
classes

# Another pre-requisites
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(oov_token='UNK', lower = True)
tokenizer.fit_on_texts(df['statement'].values)

max_len = max([len(x) for x in tokenizer.texts_to_sequences(df['statement'].values)])   # 5421

vocab_size = len(tokenizer.word_index) + 1


# Model path
PATH = './data/my_model_3.h5'

# Loaded model
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if (torch.cuda.is_available()):
    torch.cuda.empty_cache()
print(device, "\n")

model_load = torch.load(PATH, weights_only=False, map_location=torch.device(device))
model_load.eval()

import numpy as np

def transform_class(class_name):
    a = np.zeros(len(classes))
    a[classes == class_name] = 1
    return a

def rev_transform_class(class_arr):
    return classes[class_arr.argmax()]

def predict_sentiment_load(text):
    t = torch.from_numpy(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen = max_len))
    pred = model_load(t.to(device))

    return rev_transform_class(pred)


@analysis.route("/", methods=["POST"])   # main route
def result():
    sentiment = request.form['sentiment']
    if sentiment:

        result = predict_sentiment_load(sentiment)

        return render_template("result.html", data=result)
    
    else :
        return redirect('/')  # jk tdk ada text yg dimasukkan, kembalikan ke halaman home
