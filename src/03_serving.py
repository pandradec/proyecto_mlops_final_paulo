import pandas as pd
import joblib
from flask import Flask, request, jsonify

application = Flask(__name__)

# Load model and training columns
model = joblib.load("models/credit_risk_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")


@application.route("/")
def home():
    return "Credit Risk Prediction API is running."


@application.route("/predict", methods=["POST"])
def predict():
    try:
        # 1️⃣ Get JSON input
        data = request.get_json()

        # 2️⃣ Convert to DataFrame
        input_df = pd.DataFrame([data])

        # 3️⃣ Apply same encoding
        input_df = pd.get_dummies(input_df)

        # 4️⃣ Add missing columns
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # 5️⃣ Keep only training columns & correct order
        input_df = input_df[model_columns]

        # 6️⃣ Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    application.run(debug=True)