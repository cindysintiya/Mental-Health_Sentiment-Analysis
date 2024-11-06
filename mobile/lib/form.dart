import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:mental_health_gcd/provider.dart';
import 'package:mental_health_gcd/http_helper.dart';
import 'package:mental_health_gcd/prediction.dart';
import 'package:mental_health_gcd/result.dart';
import 'package:mental_health_gcd/home.dart';

class MyForm extends StatefulWidget {
  const MyForm({super.key});

  @override
  State<MyForm> createState() => _MyFormState();
}

class _MyFormState extends State<MyForm> {
  HttpHelper? helper;

  @override
  void initState() {
    helper = HttpHelper();
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    final prov = Provider.of<MyProvider>(context);

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Center(child: Text("- Mental Health Form -")),
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Container(
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.blueGrey),
                    borderRadius: BorderRadius.circular(8),
                    color: Colors.cyan[50],
                  ),
                  width: MediaQuery.of(context).size.width*0.625,
                  margin: const EdgeInsets.all(10),
                  padding: const EdgeInsets.all(10),
                  child: const Text("Hello, welcome to\nGCD - Mental Health Analysis\napp. Please fill in the form so we can analyze your mental health issue. Thank you ^^", style: TextStyle(fontSize: 16), textAlign: TextAlign.center,),
                ),
                Image.asset("assets/logo.png", width: MediaQuery.of(context).size.width*0.3,),
              ],
            ),
            Column(
              children: prov.question.entries.map<Widget>((question) =>
                Card(
                  margin: const EdgeInsets.fromLTRB(10, 10, 10, 2),
                  child: Column(
                    children: [
                      const SizedBox(height: 12,),
                      Text(
                        question.value, 
                        style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                        textAlign: TextAlign.center,
                      ),
                      const Divider(indent: 12, endIndent: 12,),
                      for (var opt in prov.model!.questions[question.key].keys.toList()) RadioListTile(
                        value: opt, 
                        groupValue: prov.form[question.key],
                        onChanged: (value) {
                          prov.form = {
                            "key": question.key,
                            "value": value
                          };
                        },
                        title: Text(opt, style: const TextStyle(fontSize: 15),),
                      ),
                      prov.form[question.key]!=Null? const SizedBox() : const Align(
                        alignment: Alignment.centerLeft,
                        child: Padding(
                          padding: EdgeInsets.only(left: 25, bottom: 15),
                          child: Text("*Required. Please choose 1 option", style: TextStyle(color: Colors.red),),
                        )
                      )
                    ],
                  ),
                )
              ).toList(),
            ),
            Card(
              margin: const EdgeInsets.all(10),
              child: Column(
                children: [
                  const SizedBox(height: 12,),
                  const Text(
                    "How do you feel these days?", 
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                    textAlign: TextAlign.center,
                  ),
                  const Divider(indent: 12, endIndent: 12,),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(22, 10, 22, 15),
                    child: Column(
                      children: [
                        DropdownButtonFormField(
                          items: [
                            const DropdownMenuItem(
                              value: "en",
                              child: Text("Default: English"),
                            ),
                            for (var lang in prov.model!.language.entries)
                              DropdownMenuItem(
                                value: lang.key,
                                child: Text(lang.value[0].toUpperCase() + lang.value.substring(1)),
                              )
                          ],
                          // items: const [
                          //   DropdownMenuItem(
                          //     value: "en",
                          //     child: Text("English"),
                          //   ),
                          //   DropdownMenuItem(
                          //     value: "id",
                          //     child: Text("Indonesian"),
                          //   )
                          // ], 
                          hint: const Text("Default: English"),
                          borderRadius: BorderRadius.circular(4),
                          decoration: InputDecoration(
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(4),
                            )
                          ),
                          onChanged: (val) {
                            prov.form = {
                              "key": "lang",
                              "value": val
                            };
                          }
                        ),
                        const SizedBox(height: 10,),
                        TextField(
                          decoration: const InputDecoration(
                            border: OutlineInputBorder(),
                            hintText: "Tell me 'bout your feeling here...",
                          ),
                          maxLines: 4,
                          onChanged: (val) => prov.sentiment = val,
                        ),
                      ],
                    ),
                  ),
                  prov.sentiment.trim().isNotEmpty? const SizedBox() : const Align(
                    alignment: Alignment.centerLeft,
                    child: Padding(
                      padding: EdgeInsets.only(left: 25, bottom: 15),
                      child: Text("*Required. Please fill in this field", style: TextStyle(color: Colors.red),),
                    )
                  )
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(15),
              child: Tooltip(
                message: prov.form.values.any((element) => element == Null,) || prov.sentiment.trim().isEmpty? "Please filled in all available field!" : "Submit your response.",
                child: FilledButton.icon(
                  onPressed: prov.form.values.any((element) => element == Null,) || prov.sentiment.trim().isEmpty? null : () async {
                    showDialog(
                      context: context, 
                      builder: (context) {
                        return AlertDialog(
                          content: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Image.asset("assets/loading.gif", width: MediaQuery.of(context).size.width*0.45,),
                              const SizedBox(height: 15,),
                              const Text("Analyzing... Please wait...", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),),
                            ],
                          ),
                          actions: [
                            Center(
                              child: ElevatedButton(
                                onPressed: () => Navigator.of(context).pop(),
                                child: const Text("Cancel"),
                              ),
                            )
                          ],
                        );
                      },
                      barrierDismissible: false,
                    );
                
                    try {
                      Prediction? result = await helper!.getPredictions(prov.host, prov.form);
                            
                      if (context.mounted) {
                        if (Navigator.of(context).canPop()) {
                          Navigator.of(context).pop();  // pop alert loading
                        }
                        Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => Result(prediction: result!,)));
                      }
                    } catch (e) {
                      if (context.mounted && Navigator.of(context).canPop()) {
                        Navigator.of(context).pop();  // pop alert loading
                        Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const HomePage()));
                      }
                    }
                  }, 
                  icon: const Icon(Icons.send_rounded),
                  label: const Text("Submit", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 17),)
                ),
              ),
            )
          ],
        ),
      )
    );
  }
}