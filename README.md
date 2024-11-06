## Pre-Requisite
- Install all packages that listed in **"requirement.txt"** file
    ```
    pip install -r requirement.txt
    ```
- Or you can run this command if you are using Anaconda
    ```
    conda install Flask Jinja2 flask-cors numpy pandas tensorflow torch scikit-learn tqdm deep-translator joblib
    ```
- Make sure that all of the packages in **"requirement.txt"** was installed on your machine
    ```
    pip list
    ```
  or
    ```
    conda list
    ```
## Run Web App or API
- If you are using Anaconda, make sure that the **(base)** is activated on your terminal (just skip if you're using pip or the base is already activated)
    ```
    C:/Users/<device_name>/anaconda3/Scripts/activate
    ```
  or
    ```
    conda activate base
    ```
- Simply type or copy this command to run the Flask app
    ```
    python app.py
    ```
## Run Mobile App
- Make sure your device are connected. You can run adb to connect your device by wireless debuging:
    ```
    adb pair <host>:<port>
    ```
    Enter pairing code: ...
    ```
    adb connect <host>:<port>
    ```
- Open terminal and go to the "mobile" folder
    ```
    cd mobile
    ```
- Simply type or copy this command to run the built app
    ```
    flutter install
    ```
    or you can run by debug (not recommended)
    ```
    flutter pub get
    ```
    ```
    flutter run
    ```