import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:mental_health_gcd/provider.dart';
import 'package:mental_health_gcd/form.dart';
import 'package:mental_health_gcd/prediction.dart';

class Result extends StatefulWidget {
  const Result({super.key, required this.prediction});

  final Prediction? prediction;

  @override
  State<Result> createState() => _ResultState();
}

class _ResultState extends State<Result> {
  @override
  Widget build(BuildContext context) {
    final prediction = widget.prediction!;
    final stress = widget.prediction!.stress;
    final sentiment = widget.prediction!.sentiment;

    final prov = Provider.of<MyProvider>(context);

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Center(child: Text("- Analysis Result -")),
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(15),
            child: Column(
              children: [
                RichText(
                  text: TextSpan(
                    text: 'So, you tell us that your occupation is ',
                    style: const TextStyle(
                      color: Colors.black,
                      fontSize: 17,
                      height: 1.8,
                    ), // Default style
                    children: <TextSpan>[
                      TextSpan(
                        text: ' ${stress["story"]["Occupation"]} ',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          backgroundColor: Colors.cyan[300],
                          color: Colors.white,
                          fontSize: 18,
                          height: 1.5,
                        ),
                      ),
                      const TextSpan(
                        text: ". \nYou ",
                      ),
                      TextSpan(
                        text: ' ${stress["story"]["Days_Indoors"]} ',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          backgroundColor: Colors.cyan[300],
                          color: Colors.white,
                          fontSize: 18,
                          height: 1.5,
                        ),
                      ),
                      const TextSpan(
                        text: ", ",
                      ),
                      TextSpan(
                        text: ' ${stress["story"]["Work_Interest"]} ',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          backgroundColor: Colors.cyan[300],
                          color: Colors.white,
                          fontSize: 18,
                          height: 1.5,
                        ),
                      ),
                      const TextSpan(
                        text: " interest in work, also ",
                      ),
                      TextSpan(
                        text: ' ${stress["story"]["Changes_Habits"]} ',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          backgroundColor: Colors.cyan[300],
                          color: Colors.white,
                          fontSize: 18,
                          height: 1.5,
                        ),
                      ),
                      const TextSpan(
                        text: " changing habits ",
                      ),
                      TextSpan(
                        text: ' ${stress["story"]["Mental_Health_History"]} ',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          backgroundColor: Colors.cyan[300],
                          color: Colors.white,
                          fontSize: 18,
                          height: 1.5,
                        ),
                      ),
                      const TextSpan(
                        text: " mental health history before. \nYou ",
                      ),
                      TextSpan(
                        text: ' ${stress["story"]["treatment"]} ',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          backgroundColor: Colors.cyan[300],
                          color: Colors.white,
                          fontSize: 18,
                          height: 1.5,
                        ),
                      ),
                      const TextSpan(
                        text: " trying to find a treatment, ",
                      ),
                      TextSpan(
                        text: ' ${stress["story"]["Coping_Struggles"]} ',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          backgroundColor: Colors.cyan[300],
                          color: Colors.white,
                          fontSize: 18,
                          height: 1.5,
                        ),
                      ),
                      const TextSpan(
                        text: " to coping it. \nYou have ",
                      ),
                      TextSpan(
                        text: ' ${stress["story"]["Mood_Swings"]} ',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          backgroundColor: Colors.cyan[300],
                          color: Colors.white,
                          fontSize: 18,
                          height: 1.5,
                        ),
                      ),
                      const TextSpan(
                        text: " mood swings, and you ",
                      ),
                      TextSpan(
                        text: ' ${stress["story"]["Social_Weakness"]} weak ',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          backgroundColor: Colors.cyan[300],
                          color: Colors.white,
                          fontSize: 18,
                          height: 1.5,
                        ),
                      ),
                      const TextSpan(
                        text: " at socializing with each other.",
                      ),
                    ],
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 15,),
                const Text(
                  "And you also tell us that",
                  style: TextStyle(
                    fontSize: 17,
                    height: 1.8,
                  ),
                ),
                Text(
                  prediction.text,
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    backgroundColor: Colors.cyan[300],
                    color: Colors.white,
                    fontSize: 18,
                    height: 1.5,
                  ),
                  textAlign: TextAlign.center,
                ),
                const Divider(height: 25,),
                Container(
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.blueGrey),
                    borderRadius: BorderRadius.circular(8),
                    color: Colors.cyan[50],
                  ),
                  margin: const EdgeInsets.fromLTRB(10, 8, 10, 0),
                  padding: const EdgeInsets.fromLTRB(15, 5, 15, 12),
                  child: Column(
                    children: [
                      RichText(
                        textAlign: TextAlign.center,
                        text: TextSpan(
                          text: 'From what you choose in the form, we predicted that\n',
                          style: const TextStyle(
                            color: Colors.black,
                            fontSize: 17,
                            height: 1.8,
                          ), // Default style
                          children: <TextSpan>[
                            const TextSpan(
                              text: "- ",
                            ),
                            TextSpan(
                              text: ' ${stress["prob"]} ',
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                backgroundColor: Colors.cyan,
                                color: Colors.white,
                                fontSize: 18,
                                height: 1.5,
                              ),
                            ),
                            const TextSpan(
                              text: " ",
                            ),
                            TextSpan(
                              text: stress["pred"]=="No"? " NOT " : ' ${stress["pred"].toUpperCase()} ',
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                backgroundColor: Colors.cyan,
                                color: Colors.white,
                                fontSize: 18,
                                height: 1.5,
                              ),
                            ),
                            const TextSpan(
                              text: " ",
                            ),
                            const TextSpan(
                              text: " STRESS ",
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                backgroundColor: Colors.cyan,
                                color: Colors.white,
                                fontSize: 18,
                                height: 1.5,
                              ),
                            ),
                            const TextSpan(
                              text: " -",
                            ),
                          ]
                        )
                      ),
                      RichText(
                        textAlign: TextAlign.center,
                        text: TextSpan(
                          text: 'And from your story, we analyze that you are\n',
                          style: const TextStyle(
                            color: Colors.black,
                            fontSize: 17,
                            height: 1.8,
                          ), // Default style
                          children: <TextSpan>[
                            const TextSpan(
                              text: "- ",
                            ),
                            TextSpan(
                              text: ' ${sentiment["prob"]} ',
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                backgroundColor: Colors.cyan,
                                color: Colors.white,
                                fontSize: 18,
                                height: 1.5,
                              ),
                            ),
                            const TextSpan(
                              text: " ",
                            ),
                            TextSpan(
                              text: ' ${sentiment["pred"].toUpperCase()} ',
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                backgroundColor: Colors.cyan,
                                color: Colors.white,
                                fontSize: 18,
                                height: 1.5,
                              ),
                            ),
                            const TextSpan(
                              text: " -",
                            ),
                          ]
                        )
                      ),
                    ],
                  ),
                ),
                Align(
                  alignment: Alignment.centerRight,
                  child: Image.asset("assets/logo.png", width: MediaQuery.of(context).size.width*0.4,),
                ),
                Padding(
                  padding: const EdgeInsets.all(5),
                  child: FilledButton.icon(
                    onPressed: () {
                      prov.resetForm();
                      Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const MyForm(),));
                    }, 
                    icon: const Icon(Icons.refresh_rounded),
                    label: const Text("Try Again", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),)
                  ),
                )
              ],
            ),
          ),
        ),
      ),
    );
  }
}