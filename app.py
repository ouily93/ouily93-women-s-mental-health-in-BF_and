

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
from pydantic import BaseModel
import xgboost as xgb
from jinja2 import Template 

app = FastAPI()

# Définition des variables d'entrée avec Pydantic
class InputVar(BaseModel):
    marie_oui_actuellement_marie: int
    Enceinte_Oui: int
    Enceinte_Non: int
    revenus_sr_commerce_formel: int
    revenus_sr_commerce_informel: int
    revenus__sr_artisanat: int 
    revenus_sr_travail_journalier: int
    revenus_sr_sans_emploi: int 
    quartier_KILWIN: int
    quartier_NIOKO2: int
    quartier_NONGHIN: int
    quartier_POLESGO: int
    quartier_TANGHIN: int
    quartier_ZONGO: int
    âge_de_la_beneficiaire: float
    poids_final_enfants: int
    TM_alimentation: int
    TM_sante_et_education: int
    sentir_fatique: int
    sentir_nerveux: int
    sentir_desespere: int
    sentir_agite_pas_rester_assis: int
    impression_tout_taiteffort: int
    se_sentir_triste: int
    senti_inutile: int

# Charger le modèle XGBoost
xgboost = joblib.load('best_est_model_xgb.pkl')

# Template HTML (comme vu plus haut)
html_template = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prédiction de Santé Mentale</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>Résultat de la Prédiction</h1>
    <div id="prediction-graph"></div>

    <script>
        // Les données de la prédiction
        var predictionValue = {{ prediction }};
        
        // Mapper les valeurs aux niveaux de dépression
        var levels = ["Pas de dépression", "Mineure", "Modérée", "Sévère"];
        var levelText = levels[predictionValue];

        // Tracer le graphique avec Plotly
        var data = [{
            x: levels,
            y: [0, 0, 0, 0],
            type: 'bar',
            marker: {
                color: ['green', 'yellow', 'orange', 'red']
            }
        }];

        // Ajuster la hauteur du niveau prédit
        data[0].y[predictionValue] = 1;

        var layout = {
            title: 'Niveau de Santé Mentale Prédit: ' + levelText,
            xaxis: {
                title: 'Niveaux',
            },
            yaxis: {
                title: 'Intensité',
                range: [0, 1.5]
            }
        };

        Plotly.newPlot('prediction-graph', data, layout);
    </script>
</body>
</html>
"""

@app.post('/predict_html', response_class=HTMLResponse)
def prediction_html(data: InputVar):
    # Transformer les données en DataFrame
    data_df = pd.json_normalize(data.dict())
    
    # Faire la prédiction
    predicted = int(xgboost.predict(data_df)[0])

    # Rendre le template HTML avec la prédiction
    template = Template(html_template)
    html_content = template.render(prediction=predicted)

    return HTMLResponse(content=html_content)

# Lancer le serveur avec Uvicorn
if __name__ == '__main__':
    uvicorn.run(app)
