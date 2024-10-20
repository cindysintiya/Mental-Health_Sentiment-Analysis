from flask import Blueprint, render_template, request, redirect

analysis = Blueprint("analysis", __name__)

import torch

classes, max_len, vocab_size, tokenizer, device, model = None, None, None, None, None, None

@analysis.route("/home")
def home_page():
    # access and save to the global scope variable so another request can still using the same previous data
    global classes, max_len, vocab_size, tokenizer, device, model

    if not model :   # cek apakah sebelumnya model sudah di-load atau belum

        import json
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        with open('./data/config.json', 'r') as f:
            config_data = json.load(f)

        # Extract variables
        classes = config_data['classes']
        max_len = config_data['max_len']
        vocab_size = config_data['vocab_size']

        # Recreate the tokenizer from its JSON string
        tokenizer = tokenizer_from_json(config_data['tokenizer'])

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
        from tensorflow.keras.preprocessing.sequence import pad_sequences

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
