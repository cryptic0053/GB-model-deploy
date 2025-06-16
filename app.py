from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import shap
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd

# Load trained model
model = joblib.load("gradient_boosting_model.pkl")

# Load training data (used only for explainability)
df = pd.read_csv("Enhanced_Kurigram_Dataset.csv").dropna()
df["Risk_Level"] = df["Risk_Level"].replace({"High": 1, "Low": 0})
X = df.drop(columns=["Risk_Level"]).values
feature_names = df.drop(columns=["Risk_Level"]).columns.tolist()

# SHAP explainer (Tree-based)
shap_explainer = shap.Explainer(model)

# LIME explainer
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

        # Prediction
        prediction = int(model.predict(features)[0])

        # SHAP explanation (top 5)
        shap_values = shap_explainer(features)
        shap_contributions = shap_values.values[0]
        top_shap = [
            {"feature": feature_names[i], "contribution": float(shap_contributions[i])}
            for i in np.argsort(np.abs(shap_contributions))[::-1][:5]
        ]

        # LIME explanation (top 5)
        lime_exp = lime_explainer.explain_instance(features[0], model.predict_proba, num_features=5)
        lime_contributions = [
            {
                "feature": str(feat).split()[0],  # optional cleanup
                "contribution": float(weight)
            }
            for feat, weight in lime_exp.as_list()
        ]

        return jsonify({
            "prediction": prediction,
            "shap_top_contributors": top_shap,
            "lime_top_contributors": lime_contributions
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
