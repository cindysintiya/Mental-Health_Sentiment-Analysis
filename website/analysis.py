from flask import Blueprint, render_template, request, redirect

analysis = Blueprint("analysis", __name__)

import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import torch

classes, max_len, vocab_size, tokenizer, device, model = None, None, None, None, None, None

@analysis.route("/home")
def home_page():
    # access and save to the global scope variable so another request can still using the same previous data
    global classes, max_len, vocab_size, tokenizer, device, model

    if not model :   # cek apakah sebelumnya model sudah di-load atau belum
        raw_data = pd.read_csv("./data/sentiments.csv")
        df = raw_data.dropna()

        classes = df['status'].unique()

        tokenizer = Tokenizer(oov_token='UNK', lower = True)
        tokenizer.fit_on_texts(df['statement'].values)

        max_len = max([len(x) for x in tokenizer.texts_to_sequences(df['statement'].values)])   # 5421
        vocab_size = len(tokenizer.word_index) + 1

        # Model path
        PATH = './data/sentiment_analysis_model.h5'

        # Checking available device (gpu/ cpu)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if (torch.cuda.is_available()):
            torch.cuda.empty_cache()

        # Load pretrained model
        model = torch.load(PATH, weights_only=False, map_location=torch.device(device))
        model.eval()

    return render_template("home.html")

@analysis.route("/result", methods=["POST"])
def result():
    sentiment = str(request.form['sentiment']).strip()
    if sentiment:
        input_tensor = torch.from_numpy(pad_sequences(tokenizer.texts_to_sequences([sentiment]), maxlen = max_len))
        prediction, probability = model(input_tensor.to(device))
        prediction = prediction.cpu().detach().numpy().argmax(axis=1).flatten()[0]

        result = {
            "pred": classes[prediction],
            "prob": "{:.2f}%".format(probability.max().item() * 100)
        }

        return render_template("result.html", text=sentiment, data=result)
    
    else :
        return redirect('/home')  # jk tdk ada text yg dimasukkan, kembalikan ke halaman home
