from flask import Blueprint, render_template, request, redirect, jsonify

analysis = Blueprint("analysis", __name__)

import torch
import json

classes, max_len, vocab_size, tokenizer, device, model_lstm = None, None, None, None, None, None
questions, model_randfor = None, None

languages = {
    'af': 'afrikaans', 'sq': 'albanian', 'ar': 'arabic', 'bn': 'bengali', 
    'bs': 'bosnian', 'bg': 'bulgarian', 'ca': 'catalan', 'zh-CN': 'chinese (simplified)', 
    'zh-TW': 'chinese (traditional)', 'hr': 'croatian', 'cs': 'czech', 'da': 'danish', 
    'nl': 'dutch', 'et': 'estonian', 'tl': 'filipino',  # , 'en': 'english'
    'fi': 'finnish', 'fr': 'french', 'de': 'german', 'el': 'greek', 
    'gu': 'gujarati', 'iw': 'hebrew', 'hi': 'hindi', 'hu': 'hungarian', 
    'is': 'iceland', 'id': 'indonesian', 'it': 'italian', 'ja': 'japanese', 
    'jw': 'javanese', 'kn': 'kannada', 'km': 'khmer', 'ko': 'korean', 
    'la': 'latin', 'lv': 'latvian', 'ms': 'malay', 'ml': 'malayalam', 
    'mr': 'marathi', 'my': 'myanmar', 'ne': 'nepali', 'no': 'norwegian', 
    'pl': 'polish', 'pt': 'portuguese', 'ro': 'romanian', 'ru': 'russian', 
    'sr': 'serbian', 'si': 'sinhala', 'sk': 'slovak', 'es': 'spanish', 
    'su': 'sundanese', 'sw': 'swahili', 'sv': 'swedish', 'ta': 'tamil', 
    'te': 'telugu', 'th': 'thai', 'tr': 'turkish', 'uk': 'ukrainian', 
    'ur': 'urdu', 'vi': 'vietnamese'
}

def get_lang(id) :
    try: 
        return languages[id].capitalize()
    except:
        return 'English'

def load_lstm_model() : 
    # access and save to the global scope variable so another request can still using the same previous data
    global classes, max_len, vocab_size, tokenizer, device, model_lstm

    if not model_lstm :   # cek apakah sebelumnya model sudah di-load atau belum

        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        with open('./data/sentiment/config.json', 'r') as f:
            config_data = json.load(f)

        # Extract variables
        classes = config_data['classes']
        max_len = config_data['max_len']
        vocab_size = config_data['vocab_size']

        # Recreate the tokenizer from its JSON string
        tokenizer = tokenizer_from_json(config_data['tokenizer'])

        # Model path
        PATH = './data/sentiment/sentiment_analysis_model.h5'

        # Checking available device (gpu/ cpu)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if (torch.cuda.is_available()):
            torch.cuda.empty_cache()

        # Load pretrained model
        model_lstm = torch.load(PATH, weights_only=False, map_location=torch.device(device))
        model_lstm.eval()

def load_randfor_model() :
    global model_randfor, questions
    
    if not model_randfor :
        import joblib

        # Model path
        PATH = './data/stress/stress_prediction_model.joblib'

        # Load pretrained model
        model_randfor = joblib.load(PATH)

        # Load feature encode
        with open('./data/stress/config.json', 'r') as f:
            questions = json.load(f)
            # Clean unused cols
            for col in ["Gender", "Country", "self_employed", "family_history", "mental_health_interview", "care_options"]:
                del questions[col]

@analysis.route("/home")
def home_page():
    load_randfor_model()
    load_lstm_model()
    return render_template("home.html", lang=languages, quest=questions)

@analysis.route("/api/load-model")
def home_api():
    load_randfor_model()
    load_lstm_model()
    return jsonify({"status": 200, "message": "Model successfully loaded!", "lang": languages})


def sentiment_analysis(text, lang) :
    from deep_translator import GoogleTranslator
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    try :
        translator = GoogleTranslator(source=lang, target='en')
        trans = translator.translate(text)
    except :
        trans = text

    input_tensor = torch.from_numpy(pad_sequences(tokenizer.texts_to_sequences([trans]), maxlen = max_len))
    prediction, probability = model_lstm(input_tensor.to(device))
    prediction = prediction.cpu().detach().numpy().argmax(axis=1).flatten()[0]

    result = {
        "pred": classes[prediction],
        "prob": "{:.2f}%".format(probability.max().item() * 100)
    }
    return result


def stress_prediction(form) : 
    import pandas as pd

    data = pd.DataFrame([form])
    del data["lang"]
    del data["sentiment"]

    for col, mapping in questions.items():
        try:
            data[col] = mapping[data[col][0]]
        except:
            continue

    predicted_result = model_randfor.predict(data)
    predicted_proba = model_randfor.predict_proba(data)

    # find the key of label where value = predicted_result[0] & get max value of proba
    result = {
        "pred": list(questions['Growing_Stress'])[int(predicted_result[0]) - 1],
        "prob": "{:.2f}%".format(predicted_proba.max().item() * 100),
        "story": {
            "Occupation": form['Occupation'],
            "treatment": "are" if form['treatment']=="Yes" else "aren't",
            "Coping_Struggles": "and still having struggles" if form['Coping_Struggles']=="Yes" else "but won't struggle",
            "Days_Indoors": ("haven't going out for " if form["Days_Indoors"]!="Go out Every day" else "") + form["Days_Indoors"].lower(),
            "Work_Interest": "really" if form['Work_Interest']=="Yes" else form['Work_Interest'].lower(),
            "Changes_Habits": "recently" if form["Changes_Habits"]=="Yes" else form["Changes_Habits"].lower(),
            "Mental_Health_History": "cause you have" if form["Mental_Health_History"]=="Yes" else ("eventhough you " + form["Mental_Health_History"].lower()) + " have",
            "Mood_Swings": form["Mood_Swings"].lower(),
            "Social_Weakness": "are" if form["Social_Weakness"]=="Yes" else form["Social_Weakness"].lower(),
        }
    }
    return result

@analysis.route("/result", methods=["POST"])
def result():
    sentiment = str(request.form['sentiment']).strip()
    try :
        lang = str(request.form['lang']).strip()
    except:
        lang = 'en'

    form = {key: val for key, val in request.form.items()}

    if sentiment:
        stress_pred = stress_prediction(form)
        sentiment_pred = sentiment_analysis(sentiment, lang)
        return render_template("result.html", lang=get_lang(lang), text=sentiment, sentiment=sentiment_pred, stress=stress_pred)
    else :
        return redirect('/home')  # jk tdk ada text yg dimasukkan, kembalikan ke halaman home

@analysis.route("/api/predict", methods=["POST"])
def api_result():
    sentiment = str(request.form['sentiment']).strip()
    try :
        lang = str(request.form['lang']).strip()
    except:
        lang = 'en'
    
    form = {key: val for key, val in request.form.items()}

    if sentiment:
        stress_pred = stress_prediction(form)
        sentiment_pred = sentiment_analysis(sentiment, lang)
        return jsonify({"status": 200, "lang": get_lang(lang), "text": sentiment, "sentiment": sentiment_pred, "stress": stress_pred})
    else :
        return jsonify({"status": 400, "message": "Text not Found"})