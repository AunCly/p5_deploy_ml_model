from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

def get_engine():
    engine = create_engine(f'postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}')
    return engine

def create_database():
    fichier_sql = Path(__file__).parent / "structure.sql"

    try:
        with open(fichier_sql, 'r', encoding='utf-8') as file:
            requetes_sql = file.read()

        with get_engine().begin() as conn:
            conn.execute(text(requetes_sql))

    except FileNotFoundError:
        print(f"Erreur : Le fichier '{fichier_sql}' est introuvable.")
    except Exception as e:
        print(f"Une erreur est survenue lors de l'exécution de la base de données : {e}")

def save_prediction(employee, result):
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO predictions (employee_data, prediction, probability) VALUES (:employee_data, :prediction, :probability)"
            ),
            {
                "employee_data": employee,
                "prediction": result['prediction'],
                "probability": result['probability']
            },
        )


def get_predictions():
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "SELECT * FROM predictions"
            ),
        )

        return [dict(row) for row in result.mappings()]
