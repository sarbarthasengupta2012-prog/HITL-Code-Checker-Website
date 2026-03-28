import json
import os
import traceback
import sys
import flask
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import dotenv

try:
    class AI:
        def __init__(self):
            self.vectorizer = CountVectorizer()
            self.clf = LogisticRegression()
            self.model = make_pipeline(self.vectorizer, self.clf)
            self.train_model()

        def train_model(self):
            if os.path.exists("data.json"):
                with open("data.json", "r") as file:
                    data = json.load(file)
                if data:
                    code_samples = [a["code"] for a in data]
                    labels = [a["label"] for a in data]
                    self.model.fit(code_samples, labels)

        def check_code(self, data):
            try:
                prediction = self.model.predict([data])
                return "This code appears clean!" if prediction[0] == 1 else "This code has bad logic."
            except:
                return "Model not trained yet."


    app = Flask(__name__)
    socketio = SocketIO(app)
    model = AI()

    dotenv.load_dotenv(".env")


    @app.route("/")
    def home():
        return render_template("index.html")


    @app.route("/check_password", methods=["POST"])
    def checkPanel():
        data = flask.request.json
        user_input = data.get("password")

        if user_input == os.getenv("ADMIN_SECRET"):
            return flask.jsonify({"status": "success"})
        else:
            return flask.jsonify({"status": "fail"}), 403


    @app.route("/admin")
    def admin():
        return render_template("admin.html")


    @socketio.on("codesocket")
    def handle_code(data):
        result = model.check_code(data)
        emit("res", result)


    @socketio.on("adminsocket")
    def forward_to_admin(data):
        emit("adminsocket", data, broadcast=True)


    @socketio.on("save_to_json")
    def save_and_retrain(data):
        file_path = "data.json"
        current_data = []

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                current_data = json.load(f)

        current_data.append(data)

        with open(file_path, "w") as f:
            json.dump(current_data, f, indent=4)

        model.train_model()


    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 5000))
        socketio.run(app, host="0.0.0.0", port=port)
except Exception as e:
    print(traceback.format_exc())
    sys.exit(1)