import os
import traceback
import sys
import flask
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import dotenv
import pymongo

# Load environment variables
dotenv.load_dotenv(".env")

try:
    # --- MongoDB Setup ---
    client = pymongo.MongoClient(os.environ.get("MONGO"))
    db = client["codechecker"]
    training_collection = db["training"]
    queries_collection = db["queries"]


    class AI:
        def __init__(self):
            self.vectorizer = CountVectorizer()
            self.clf = LogisticRegression()
            self.model = make_pipeline(self.vectorizer, self.clf)
            self.train_model()

        def train_model(self):
            # Pull all training data from MongoDB
            cursor = training_collection.find({})
            data = list(cursor)

            if len(data) > 0:
                code_samples = [item["code"] for item in data]
                labels = [item["label"] for item in data]
                try:
                    self.model.fit(code_samples, labels)
                    print(f"Model trained on {len(data)} samples from MongoDB.")
                except Exception as e:
                    print(f"Training failed: {e}")

        def check_code(self, code_text):
            try:
                prediction = self.model.predict([code_text])
                return "This code appears clean!" if prediction[0] == 1 else "This code has bad logic."
            except:
                return "Model not trained yet."


    app = Flask(__name__)
    socketio = SocketIO(app)
    model = AI()


    @app.route("/")
    def home():
        return render_template("index.html")


    @app.route("/check_password", methods=["POST"])
    def check_panel():
        data = request.json
        user_input = data.get("password")
        if user_input == os.getenv("ADMIN_SECRET"):
            return jsonify({"status": "success"})
        return jsonify({"status": "fail"}), 403


    @app.route("/admin")
    def admin():
        return render_template("admin.html")


    @socketio.on("codesocket")
    def handle_code(data):
        result = model.check_code(data)
        queries_collection.insert_one({"code": data, "result": result})
        emit("res", result)


    @socketio.on("adminsocket")
    def forward_to_admin(data):
        emit("adminsocket", data, broadcast=True)
        queries_collection.insert_one(data)


    @socketio.on("save_to_json")
    def save_and_retrain(data):
        if "code" in data and "label" in data:
            training_collection.insert_one(data)
            model.train_model()
            print("New data saved to MongoDB and model retrained.")


    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 5000))
        socketio.run(app, host="0.0.0.0", port=port)

except Exception:
    print(traceback.format_exc())
    sys.exit(1)