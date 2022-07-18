import json
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

app = Flask(__name__)


load_dotenv(".env")


@app.route("/")
def api_hello_world():
    return "Hello from Flask!"


@app.route("/env", methods=["GET", "POST"])
def api_env():
    djson = {}
    if request.method == "POST":
        djson = request.get_json()
    djson["env"] = os.environ["AAA"]
    return jsonify(djson)


@app.route("/message", methods=["GET", "POST"])
def api_message():
    return json.dumps({"message": "Hello from Flask!"})
