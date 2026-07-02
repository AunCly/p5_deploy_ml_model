import pandas as pd

from employee import EmployeeData

def format_data_for_ml_model(employee: EmployeeData) -> pd.DataFrame:

    data = pd.DataFrame([employee.model_dump()])

    data['genre'] = data['genre'].map({'F': 0, 'M': 1})
    data['a_quitte_l_entreprise'] = data['a_quitte_l_entreprise'].map({'Non': 0, 'Oui': 1})
    data['heure_supplementaires'] = data['heure_supplementaires'].map({'Non': 0, 'Oui': 1})
    data['frequence_deplacement'] = data['frequence_deplacement'].map({'Aucun': 1, 'Occasionnel': 1.5, 'Frequent': 2})

    cols_to_remove = [
        'id_employee',
        'nombre_heures_travailless',
        'augementation_salaire_precedente',
        'nombre_employee_sous_responsabilite',
        'ayant_enfants'
    ]

    data.drop(columns=cols_to_remove, inplace=True)

    return data