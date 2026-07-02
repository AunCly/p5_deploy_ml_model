import pandas as pd
from pydantic import BaseModel, Field, model_validator

class EmployeeData(BaseModel):
    id_employee: int
    age: int
    genre: str
    revenu_mensuel: float
    statut_marital: str
    departement: str
    poste: str
    nombre_experiences_precedentes: int
    nombre_heures_travailless: int
    annee_experience_totale: int
    annees_dans_l_entreprise: int
    annees_dans_le_poste_actuel: int
    satisfaction_employee_environnement: int
    note_evaluation_precedente: int
    niveau_hierarchique_poste: int
    satisfaction_employee_nature_travail: int
    satisfaction_employee_equipe: int
    satisfaction_employee_equilibre_pro_perso: int
    note_evaluation_actuelle: int
    heure_supplementaires: str
    augementation_salaire_precedente: str
    a_quitte_l_entreprise: str
    nombre_participation_pee: int
    nb_formations_suivies: int
    nombre_employee_sous_responsabilite: int
    distance_domicile_travail: int
    niveau_education: int
    domaine_etude: str
    ayant_enfants: str
    frequence_deplacement: str
    annees_depuis_la_derniere_promotion: int
    annes_sous_responsable_actuel: int