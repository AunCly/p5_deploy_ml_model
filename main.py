import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from employee import EmployeeData
from database import database_manager as database
from database import seeding as seeding
from contextlib import asynccontextmanager

from models.predictor import AttritionPredictor

load_dotenv()

api_key_header = APIKeyHeader(name="x-api-key")
predictor = AttritionPredictor()

def api_predict(employee: EmployeeData):
    result = predictor.predict(employee)

    try:
        if os.getenv('ENVIRONMENT') != 'production':
            database.save_prediction(employee.model_dump_json(), result)
    except Exception as e:
        print(f"Erreur lors de l'enregistrement de la prédiction : {e}")

    return {
        "employee_id": employee.id_employee,
        "prediction": result['prediction'],
        "probability": result['probability'],
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv('ENVIRONMENT') != 'production':
        database.create_database()
        seeding.seed()
    yield

app = FastAPI(
    lifespan=lifespan,
    title="API de prédiction de turnover",
    description="Cette API permet de prédire le turnover des employés en utilisant un modèle de machine learning. Elle offre des endpoints pour effectuer des prédictions individuelles ou par lot, ainsi que pour récupérer l'historique des prédictions.",
    version="0.0.1",
    openapi_tags=[
        {
            "name": "Prédiction de turnover",
            "description": "Endpoints pour effectuer des prédictions de turnover et récupérer l'historique des prédictions."
        }
    ],
    contact={
        "email": "aurelien.clugery@gmail.com",
    }
)

async def verify_api_key(api_key: str = Security(api_key_header)):

    key = os.getenv('API_KEY')

    if not key or api_key != key:
        raise HTTPException(status_code=403)

    return api_key

@app.get(
    "/health",
    summary="Vérifier la santé de l'API",
    responses={
        200: {"description": "Le serveur de l'API fonctionne correctement"},
    }
)
def health_endpoint():
    return {"message": "Alive !"}

@app.post(
    "/predict",
    dependencies=[Depends(verify_api_key)],
    summary="Effectuer une prédiction de turnover",
    description="Envoie les données d'un employé au modèle pour prédire s'il risque de quitter l'entreprise.",
    responses={
        200: {"description": "Prédiction réussie"},
        401: {"description": "Clé API manquante ou invalide"},
        422: {"description": "Données d'employé invalides (erreur de validation Pydantic)"}
    }
)
def predict_endpoint(employee: EmployeeData):

    prediction = api_predict(employee)

    return prediction

@app.post(
    "/predict/batch",
    dependencies=[Depends(verify_api_key)],
    summary="Effectuer une série de prédiction de turnover",
    description="Permet de prédire le turnover d'un ensemble d'employés.",
    responses={
        200: {"description": "Prédictions réussie"},
        401: {"description": "Clé API manquante ou invalide"},
        422: {"description": "Données d'employé invalides (erreur de validation Pydantic)"}
    }
)
def predict_bach_endpoint(employees: list[EmployeeData]):

    predictions = []

    for employee in employees:
        prediction = api_predict(employee)
        predictions.append(prediction)

    return predictions

@app.get(
    "/predict/history",
    dependencies=[Depends(verify_api_key)],
    summary="Récupérer l'historique des prédictions",
    description="Permet de récupérer l'historique des prédictions effectuées par l'API.",
    responses={
        200: {"description": "Historique récupéré avec succès"},
        401: {"description": "Clé API manquante ou invalide"},
    }
)
def history_endpoint():
    database_predictions = database.get_predictions()
    predictions = []
    for prediction in database_predictions:
        predictions.append(prediction)

    return predictions



