# app_flask.py
from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd
app = Flask(__name__)

# Charger le modèle XGBoost
xgboost = joblib.load('best_est_model_xgb.pkl')

# Route pour servir la page HTML
@app.route('/')
def index():
    return render_template('index.html')

# Route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    data_df = pd.json_normalize(data) 

    # Prédiction
    predicted = xgboost.predict(data_df)[0] 

    # Mapper les valeurs de prédiction aux descriptions
    depression_levels = {0: "pas de dépression", 1: "mineure", 2: "modérée", 3: "sévère"}
    result = {
        "message": f"Vous avez un état de santé mentale de type {depression_levels.get(predicted, 'inconnu')}.",
        "prediction": int(predicted)
    }
    return jsonify(result)
if __name__ == '__main__':
    app.run(debug=True)
