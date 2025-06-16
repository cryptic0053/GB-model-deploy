from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import shap
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import re
from collections import OrderedDict  # to keep JSON key order

# Load model
model = joblib.load("gradient_boosting_model.pkl")

# Load training data (for LIME & SHAP explainers)
df = pd.read_csv("Enhanced_Kurigram_Dataset.csv").dropna()
df["Risk_Level"] = df["Risk_Level"].replace({"High": 1, "Low": 0})
X = df.drop(columns=["Risk_Level"]).values
feature_names = df.drop(columns=["Risk_Level"]).columns.tolist()

# SHAP Explainer (TreeExplainer for Gradient Boosting)
shap_explainer = shap.Explainer(model)

# LIME Explainer
lime_explainer = LimeTabularExplainer(
    training_data=X,
    feature_names=feature_names,
    class_names=["Low Risk", "High Risk"],
    mode="classification"
)

# Flask setup
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Gradient Boosting API with SHAP + LIME is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        # Raw prediction
        raw_prediction = int(model.predict(features)[0])
        prediction_label = "High Risk" if raw_prediction == 1 else "Low Risk"

        # SHAP Explanation (Top 5)
        shap_values = shap_explainer(features)
        shap_contributions = shap_values.values[0]
        top_shap = [
            {"feature": feature_names[i], "contribution": float(shap_contributions[i])}
            for i in np.argsort(np.abs(shap_contributions))[::-1][:5]
        ]

        # LIME Explanation (Top 5, cleaned names)
        lime_exp = lime_explainer.explain_instance(features[0], model.predict_proba, num_features=5)
        lime_contributions = []
        for feat, weight in lime_exp.as_list():
            match = re.match(r"([^\s><=]+)", feat)
            clean_feature = match.group(1) if match else str(feat)
            lime_contributions.append({
                "feature": clean_feature,
                "contribution": float(weight)
            })

        # Return in custom order using OrderedDict
        result = OrderedDict()
        result["prediction"] = prediction_label
        result["shap_top_contributors"] = top_shap
        result["lime_top_contributors"] = lime_contributions

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
