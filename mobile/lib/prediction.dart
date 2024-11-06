class Prediction {
  final String lang;
  final String text;
  final Map<String, dynamic> sentiment;
  final Map<String, dynamic> stress;

  Prediction(this.lang, this.text, this.sentiment, this.stress);

  factory Prediction.fromJson(Map<String, dynamic> parsedJson) {
    final lang = parsedJson['lang'] as String;
    final text = parsedJson['text'] as String;
    final sentiment = parsedJson['sentiment'] as Map<String, dynamic>;
    final stress = parsedJson['stress'] as Map<String, dynamic>;

    return Prediction(lang, text, sentiment, stress);
  }
}