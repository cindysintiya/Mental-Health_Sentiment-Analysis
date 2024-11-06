import 'package:flutter/material.dart';
import 'package:mental_health_gcd/load_model.dart';

class MyProvider extends ChangeNotifier {
  TextEditingController hostIp = TextEditingController(text: ""); // 192.168.1.3
  String _host = ""; // 192.168.1.3
  String get host => _host;

  set host(val) {
    _host = val;
    notifyListeners();
  }

  LoadModel? _model;
  LoadModel? get model => _model;

  set model(val) {
    _model = val;
    // notifyListeners();
  }

  final question = {
    "Occupation": "What is your occupation?",
    "treatment": "Have you ever find a treatment for your mental health issue?",
    "Days_Indoors": "How long have you stayed at home/ not going out these days?",
    "Changes_Habits": "Do you feel a change in your habits lately?",
    "Mental_Health_History": "Do you have mental health history before?",
    "Mood_Swings": "How bad is your mood swings?",
    "Coping_Struggles": "Are you struggle to coping your mental health issue?",
    "Work_Interest": "Do you really interest in your occupation/ work?",
    "Social_Weakness": "Do you have trouble socializing with others?",
  };

  Map _form = {
    "Occupation": Null,
    "treatment": Null,
    "Days_Indoors": Null,
    "Changes_Habits": Null,
    "Mental_Health_History": Null,
    "Mood_Swings": Null,
    "Coping_Struggles": Null,
    "Work_Interest": Null,
    "Social_Weakness": Null,
    "sentiment": Null,
    "lang": "en",
  };
  Map get form => _form;

  set form(val) {
    _form[val["key"]] = val["value"];
    notifyListeners();
  }

  resetForm() {
    _form = {
      "Occupation": Null,
      "treatment": Null,
      "Days_Indoors": Null,
      "Changes_Habits": Null,
      "Mental_Health_History": Null,
      "Mood_Swings": Null,
      "Coping_Struggles": Null,
      "Work_Interest": Null,
      "Social_Weakness": Null,
      "sentiment": Null,
      "lang": "en",
    };
    notifyListeners();
  }

  String _sentiment = "";
  String get sentiment => _sentiment;

  set sentiment(val) {
    _sentiment = val;
    _form["sentiment"] = val;
    notifyListeners();
  }
}