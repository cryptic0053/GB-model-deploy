from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Load model
model = joblib.load("gradient_boosting_model.pkl")

# Load training data
df = pd.read_csv("Enhanced_Kurigram_Dataset.csv").dropna()
df["Risk_Level"] = df["Risk_Level"].replace({"High": 1, "Low": 0})
X = df.drop(columns=["Risk_Level"]).values
feature_names = df.drop(columns=["Risk_Level"]).columns.tolist()

# LIME Explainer
lime_explainer = LimeTabularExplainer(
    training_data=X,
    feature_names=feature_names,
    class_names=["Low Risk", "High Risk"],
    mode="classification"
)

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "\u2705 Gradient Boosting API with LIME is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        # Predict
        raw_prediction = int(model.predict(features)[0])
        prediction_label = "High Risk" if raw_prediction == 1 else "Low Risk"

        # LIME Explanation (All features)
        lime_exp = lime_explainer.explain_instance(features[0], model.predict_proba, num_features=len(feature_names))
        lime_contributions = []
        feature_weights = {}

        for feat, weight in lime_exp.as_list():
            clean_feature = None
            for f in feature_names:
                if feat.startswith(f) or f in feat:
                    clean_feature = f
                    break
            if clean_feature is None:
                clean_feature = str(feat)

            lime_contributions.append({
                "feature": clean_feature,
                "contribution": float(weight)
            })
            feature_weights[clean_feature] = float(weight)

        # Plot bar chart
        plt.figure(figsize=(10, 6))
        plt.barh(list(feature_weights.keys()), list(feature_weights.values()), color="skyblue")
        plt.xlabel("Contribution Weight")
        plt.title("LIME Feature Contributions")
        plt.tight_layout()

        # Encode chart as base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        buffer.close()

        # Final response
        result = {
            "prediction": prediction_label,
            "lime_contributors": lime_contributions,
            "lime_chart_base64": chart_base64
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
