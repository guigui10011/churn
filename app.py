from flask import Flask, request, jsonify, render_template

import joblib
import numpy as np

model = joblib.load("data/model_sklearn.pkl")

app = Flask(__name__)


# Route principale pour afficher l'interface utilisateur
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        age = float(request.form["Age"])
        account_manager = float(request.form["Account_Manager"])
        years = float(request.form["Years"])
        num_sites = float(request.form["Num_Sites"])

        # # Extract input values
        # age = data.get("Age", None)
        # account_manager = data.get("Account_Manager", None)
        # years = data.get("Years", None)
        # num_sites = data.get("Num_Sites", None)

        input_data = np.array([[age, account_manager, years, num_sites]])

        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data).max()  # Get the highest probability

        result = "Churn, perdu" if prediction[0] == 1 else "Il reste"

        return jsonify({
            "prediction": result,
            "confidence": f"{probability:.2f}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000,)