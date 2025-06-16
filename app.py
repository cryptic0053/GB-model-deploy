from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Load the trained Gradient Boosting model
model = joblib.load("gradient_boosting_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Gradient Boosting API is up and running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app on 0.0.0.0 to expose it externally on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
