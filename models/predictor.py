from pathlib import Path

import joblib
import os

from dotenv import load_dotenv

from employee import EmployeeData
from . import preprocessing

class AttritionPredictor:
    def __init__(self, model_name: str = "classification_model", preprocessor_name: str = "classification_preprocessor"):

        load_dotenv()
        base_dir = Path(__file__).resolve().parent.parent

        self.model_path = base_dir / f"models/compiled/{model_name}.joblib"
        self.preprocessor_path = base_dir / f"models/compiled/{preprocessor_name}.joblib"

        self.preprocessor = self._load(self.preprocessor_path)
        self.model = self._load(self.model_path)

    def _load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artefact introuvable : {path}")
        return joblib.load(path)

    def predict(self, employee_data: "EmployeeData") -> dict:
        df_input = preprocessing.format_data_for_ml_model(employee_data)

        X = self.preprocessor.transform(df_input)

        prediction = self.model.predict(X)
        probability = self.model.predict_proba(X)

        return {
            "prediction": int(prediction[0]),
            "probability": float(probability[0][1]),
        }