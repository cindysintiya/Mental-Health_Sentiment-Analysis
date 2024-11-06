class LoadModel {
  final Map<String, String> language;
  final Map<dynamic, dynamic> questions;

  LoadModel(this.language, this.questions);

  factory LoadModel.fromJson(Map<String, dynamic> parsedJson) {
    final language = Map<String, String>.from(parsedJson['lang']);
    final questions = Map<dynamic, dynamic>.from(parsedJson['quest']);

    return LoadModel(language, questions);
  }
}