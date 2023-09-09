from model import Model
from flask import Flask, request, render_template

model = Model()
model.load()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api")
def api():
    prompt = request.args.get("prompt")
    response = model.predict(prompt)

    json = {"input": prompt, "response": response}
    return json
