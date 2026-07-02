import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi.testclient import TestClient
from main import app
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
load_dotenv()

client = TestClient(app)

def get_random_employee():
    data_sirh = pd.read_csv(base_dir / 'data/raw/extrait_sirh.csv', sep=',', na_values=[''], quotechar='"')
    data_eval = pd.read_csv(base_dir / 'data/raw/extrait_eval.csv', sep=',', na_values=[''], quotechar='"')
    data_sondage = pd.read_csv(base_dir / 'data/raw/extrait_sondage.csv', sep=',', na_values=[''], quotechar='"')

    data_eval['id_employee'] = data_eval['eval_number'].apply(lambda x : x.split('E_')[1])
    data_eval['id_employee'] = data_eval['id_employee'].astype(int)

    data_sondage['id_employee'] = data_sondage['code_sondage']
    data_sondage['id_employee'] = data_sondage['id_employee'].astype(int)

    # Fusion des fichiers
    data = data_sirh.merge(data_eval, how='inner')
    data = data.merge(data_sondage, how='inner')

    sample_df = data.sample(1)

    data_dict = sample_df.to_dict(orient='records')[0]

    return data_dict

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "Alive !"}

def test_predict_with_wrong_api_key():
    response = client.post("/predict", headers={"x-api-key": "foo"})
    assert response.status_code == 403
    assert response.json() == {"detail": "Forbidden"}

def test_predict():
    test_employee = get_random_employee()
    response = client.post("/predict", json=test_employee, headers={"x-api-key": os.getenv('API_KEY')})

    response_keys_needed = {"probability", "prediction", "employee_id"}

    assert response.status_code == 200
    assert response_keys_needed <= response.json().keys()
    assert response_keys_needed == response.json().keys()
