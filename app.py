from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the model
model = joblib.load("gradient_boosting_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Gradient Boosting API is up!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
