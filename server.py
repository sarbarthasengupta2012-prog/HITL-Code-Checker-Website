import os
import traceback
import sys
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

# Database Setup
client = pymongo.MongoClient(os.environ.get("MONGO"))
db = client["codechecker"]
training_collection = db["training"]
queries_collection = db["queries"]

class AI:
    def __init__(self):
        try:
            self.model = joblib.load("model.pkl")
            print("AI loaded from disk.")
        except:
            print("Creating new model...")
            self.model = make_pipeline(CountVectorizer(), LogisticRegression())
            self.train_model()

    def train_model(self):
        cursor = training_collection.find({})
        data = list(cursor)
        labels = [item["label"] for item in data]
        if len(set(labels)) > 1:
            code_samples = [item["code"] for item in data]
            self.model.fit(code_samples, labels)
            joblib.dump(self.model, "model.pkl")
            print(f"Model trained & saved on {len(data)} samples.")
        else:
            print("Not enough diverse data to train yet (need both clean and messy).")

    def check_code(self, code_text):
        try:
            prediction = self.model.predict([code_text])
            return "This code appears clean!" if prediction[0] == 1 else "This code has bad logic."
        except Exception as e:
            return "Model is still learning..."

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
    queries_collection.insert_one(data)

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