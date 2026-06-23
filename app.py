
from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import os
import json
from crop_info import crop_info

app = Flask(__name__)

# Load shared resources
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/target_mapping.pkl', 'rb') as f:
    target_mapping = pickle.load(f)

inv_target_mapping = {v: k for k, v in target_mapping.items()}

MODEL_FILES = {
    'knn': 'knn.pkl',
    'svm_linear': 'svm_linear.pkl',
    'svm_rbf': 'svm_rbf.pkl',
    'svm_poly': 'svm_poly.pkl',
    'decision_tree': 'decision_tree.pkl',
    'random_forest': 'random_forest.pkl',
    'gradient_boosting': 'gradient_boosting.pkl'
}

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("predict.html", models=MODEL_FILES.keys(), selected_model='knn')

    try:
        model_name = request.form['model']
        model_path = os.path.join("model", MODEL_FILES[model_name])
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        input_data = {f: request.form[f] for f in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']}
        values = [float(input_data[f]) for f in input_data]
        input_scaled = scaler.transform([values])
        prediction_code = model.predict(input_scaled)[0]
        prediction_label = target_mapping[prediction_code]

        info = crop_info.get(prediction_label.lower(), {})
        return render_template(
            "predict.html",
            models=MODEL_FILES.keys(),
            prediction=prediction_label,
            selected_model=model_name,
            crop_image=info.get("image"),
            crop_description=info.get("description"),
            input_values=input_data
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"

@app.route('/models')
def models_dashboard():
    try:
        with open("static/charts/model_insights.json", "r") as f:
            models = json.load(f)
        return render_template("models.html", models=models)
    except Exception as e:
        return f"Failed to load insights: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
