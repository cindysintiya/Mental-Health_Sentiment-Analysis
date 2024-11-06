import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:mental_health_gcd/provider.dart';
import 'package:mental_health_gcd/http_helper.dart';
import 'package:mental_health_gcd/load_model.dart';
import 'package:mental_health_gcd/form.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
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
        title: const Center(child: Text("Mental Health - GCD", style: TextStyle(fontWeight: FontWeight.bold),)),
        actions: [
          IconButton(
            onPressed: () {
              showDialog(
                context: context, 
                builder: (context) {
                  return AlertDialog(
                    title: const Text("Host IP Address"),
                    content: TextField(
                      controller: prov.hostIp,
                      decoration: const InputDecoration(hintText: "Enter Host IP Address"),
                    ),
                    actions: [
                      FilledButton(
                        onPressed: () {
                          prov.host = prov.hostIp.text;
                          Navigator.pop(context);
                        }, 
                        child: const Text("Save and Try Again",),
                      ),
                    ],
                  );
                },
                barrierDismissible: false,
              );
            }, 
            icon: const Icon(Icons.settings_outlined),
            tooltip: "Setting",
          )
        ],
      ),
      body: Center(
        child: SingleChildScrollView(
          child: prov.host.isEmpty? Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Image.asset("assets/logo.png", width: MediaQuery.of(context).size.width*0.7,),
                const Padding(
                  padding: EdgeInsets.fromLTRB(20, 15, 20, 10),
                  child: Text("Please fill the Host IP from setting for backend API fetching.", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 17), textAlign: TextAlign.center,),
                ),
              ],
            ) : FutureBuilder(
            future: helper?.loadModel(prov.host),
            builder: (context, snapshot) {
              if (snapshot.hasData && snapshot.connectionState == ConnectionState.done) {
                prov.model = snapshot.data as LoadModel;
                WidgetsBinding.instance.addPostFrameCallback((_) {
                  Future.delayed(const Duration(milliseconds: 700), () {
                    if (context.mounted) {
                      if (Navigator.of(context).canPop()) {
                        Navigator.of(context).pop();  // pop alert input host ip
                      }
                      Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const MyForm()));
                    }
                  });
                });
              } else if (snapshot.hasError && snapshot.connectionState == ConnectionState.done) {
                return Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Image.asset("assets/logo.png", width: MediaQuery.of(context).size.width*0.7,),
                    const Padding(
                      padding: EdgeInsets.fromLTRB(20, 15, 20, 10),
                      child: Text("Error!!! Server off, wrong host ip, or maybe no internet connection. Please check on either 1 of it or change the Host IP on setting.", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 17), textAlign: TextAlign.center,),
                    ),
                  ],
                );
              }
              return Column(
                children: [
                  Image.asset("assets/loading.gif", width: MediaQuery.of(context).size.width,),
                  const Text(
                    "We're loading Model for you! Please Wait...",
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 20),
                    textAlign: TextAlign.center,
                  ),
                ],
              );
            }
          ),
        ),
      ),
    );
  }
}