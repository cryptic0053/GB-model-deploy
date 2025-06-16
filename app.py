from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import io
import base64

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
    return "✅ Gradient Boosting API with LIME Visualization is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        # Prediction
        raw_prediction = int(model.predict(features)[0])
        prediction_label = "High Risk" if raw_prediction == 1 else "Low Risk"

        # LIME Explanation (all features)
        lime_exp = lime_explainer.explain_instance(features[0], model.predict_proba, num_features=len(feature_names))
        lime_contributions = []
        weights_dict = dict(lime_exp.as_list())

        for f in feature_names:
            matched = next((feat for feat in weights_dict if f in feat), None)
            contribution = weights_dict.get(matched, 0.0)
            lime_contributions.append({
                "feature": f,
                "contribution": float(contribution)
            })

        # Plot chart
        fig, ax = plt.subplots()
        feat_names = [item["feature"] for item in lime_contributions]
        contribs = [item["contribution"] for item in lime_contributions]
        colors = ['green' if c > 0 else 'red' for c in contribs]

        ax.barh(feat_names, contribs, color=colors)
        ax.set_xlabel("Contribution to Prediction")
        ax.set_title("LIME Explanation")
        plt.tight_layout()

        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close()

        return jsonify({
            "prediction": prediction_label,
            "lime_contributions": lime_contributions,
            "lime_chart_base64": image_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
