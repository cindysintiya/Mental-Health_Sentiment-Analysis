from flask import Blueprint, render_template

loading = Blueprint("loading", __name__)

@loading.route("/")
def loading_page():
    return render_template("loading.html")