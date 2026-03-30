import os
import joblib
import dotenv
import pymongo
import json
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from bson import json_util
from bson.objectid import ObjectId

dotenv.load_dotenv(".env")

client = pymongo.MongoClient(os.environ.get("MONGO"))
db = client["codechecker"]
training_collection = db["training"]
queries_collection = db["queries"]


class AI:
    def __init__(self):
        self.model_path = "model.pkl"
        try:
            self.model = joblib.load(self.model_path)
            print("AI loaded from "+self.model_path)
        except:
            print("No model found, making new pkl file.")
            self.model = make_pipeline(CountVectorizer(), LogisticRegression())
            self.train_model()

    def train_model(self):
        cursor = training_collection.find({})
        data = list(cursor)

        if len(data) < 2:
            print("Training aborted: Need at least 2 samples to train.")
            return
        labels = []
        code_samples = []

        for item in data:
            labels.append(int(item["label"]))
            code_samples.append(str(item["code"]))
        if len(set(labels)) < 2:
            print("Needs more variety in the db.")
            return
        self.model.fit(code_samples, labels)
        joblib.dump(self.model, self.model_path)
        print(f"Model successfully trained!")
    def check_code(self, code_text):
        if not code_text or str(code_text).strip() == "":
            return "Please provide some code to check."
        try:
            prediction = self.model.predict([str(code_text)])[0]
            prob = self.model.predict_proba([str(code_text)])[0]
            strA = str(int(prob[1]*100))+"%"
            if int(prediction) == 1:
                return ("This code appears clean!"
                        "\nConfidence score: "+strA)
            else:
                return ("This code is spaghetti."
                        "\nConfidence score: " +strA)
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error!"

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
model = AI()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/check_password", methods=["POST"])
def check_panel():
    data = request.json
    if data.get("password") == os.getenv("ADMIN_SECRET"):
        return jsonify({"status": "success"})
    return jsonify({"status": "fail"}), 403

@socketio.on("codesocket")
def handle_code_check(code_text):
    result = model.check_code(code_text)
    emit("res", result)

@socketio.on("adminsocket")
def handle_admin_report(data):
    ae = queries_collection.insert_one(data)
    data["_id"] = str(ae.inserted_id)
    emit("updateLogs",data, broadcast=True)

@socketio.on("get_info")
def handle_get_info():
    docs = list(queries_collection.find())
    clean_data = json.loads(json_util.dumps(docs))
    emit("get_info", clean_data)

@socketio.on("save_to_json")
def save_and_retrain(data):
    training_collection.insert_one({"code": data["code"], "label": data["label"]})
    if "id" in data:
        queries_collection.delete_one({"_id": ObjectId(data["id"])})
    model.train_model()

@socketio.on("delete_query")
def delete_query(data):
    queries_collection.delete_one({"_id": ObjectId(data["id"])})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)