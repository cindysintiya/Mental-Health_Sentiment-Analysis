import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:mental_health_gcd/load_model.dart';
import 'package:mental_health_gcd/prediction.dart';

class HttpHelper {
  // final String _host = "192.168.1.3";

  Future<LoadModel?> loadModel(host) async {    // pengambilan data melalui web service
    var url = Uri.parse('http://$host:5000/api/load-model');
    http.Response result = await http.get(url, headers: { "Access-Control-Allow-Origin": "*" }).timeout(
      const Duration(seconds: 45),  // timeout after 45s trying
      onTimeout: () {
        throw TimeoutException("Connection timed out, please try again later.");
      },);

    if (result.statusCode == 200) {
      final Map<String, dynamic> jsonResponse = json.decode(result.body) as Map<String, dynamic>;
      // {"status": 200, "message": "Model successfully loaded!", "lang": languages, "quest": questions}
      LoadModel data = LoadModel.fromJson(jsonResponse);
      return data;
    }
    return null;
  }

  Future<Prediction?> getPredictions(host, form) async {    // pengambilan data melalui web service
    var url = Uri.parse('http://$host:5000/api/predict');
    http.Response result = await http.post(url, body: form);

    if (result.statusCode == 200) {
      final jsonResponse = json.decode(result.body);
      // {"status": 200, "lang": get_lang(lang), "text": sentiment, "sentiment": sentiment_pred, "stress": stress_pred}
      Prediction prediction = Prediction.fromJson(jsonResponse);
      return prediction;
    }
    return null;
  }
}