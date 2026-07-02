import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TechNovaFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.quartiles_par_poste_ = {}

    def fit(self, X, y=None):
        groupes = X.groupby('poste')['revenu_mensuel']
        for poste, valeurs in groupes:
            self.quartiles_par_poste_[poste] = valeurs.quantile([0.25, 0.5, 0.75]).values
        return self

    def transform(self, X):
        X_out = X.copy()

        # 1. Ratios d'ancienneté et d'expérience
        X_out['ratio_anciennete_total_experence'] = np.where(
            X_out['annee_experience_totale'] > 0,
            X_out['annees_dans_l_entreprise'] / X_out['annee_experience_totale'],
            0
        )

        X_out['ratio_poste_actuel_anciennete'] = np.where(
            X_out['annees_dans_l_entreprise'] > 0,
            X_out['annees_dans_le_poste_actuel'] / X_out['annees_dans_l_entreprise'],
            X_out['annees_dans_le_poste_actuel']
        )

        # Score de pénibilité du transport
        X_out['score_penibilite_transport'] = X_out['distance_domicile_travail'] * X_out['frequence_deplacement']

        # Score bien-être
        colonnes_satisfaction = [
            'satisfaction_employee_environnement',
            'satisfaction_employee_nature_travail',
            'satisfaction_employee_equipe',
            'satisfaction_employee_equilibre_pro_perso'
        ]
        X_out['score_bien_etre'] = X_out[colonnes_satisfaction].mean(axis=1)

        # Scores de performance et d'évolution
        colonnes_evaluation = ['note_evaluation_precedente', 'note_evaluation_actuelle']
        X_out['score_performance'] = X_out[colonnes_evaluation].mean(axis=1)
        X_out['evolution_performance'] = X_out['note_evaluation_actuelle'] - X_out['note_evaluation_precedente']

        # Evolution et stabilité des anciennes expériences
        X_out['evolution_hierarchie_score'] = np.where(
            X_out['annee_experience_totale'] > 0,
            X_out['niveau_hierarchique_poste'] / X_out['annee_experience_totale'],
            X_out['niveau_hierarchique_poste']
        )

        X_out['annee_par_experience'] = np.where(
            X_out['nombre_experiences_precedentes'] > 0,
            X_out['annee_experience_totale'] / X_out['nombre_experiences_precedentes'],
            X_out['annee_experience_totale']
        )

        def determiner_quartile(row):
            poste = row['poste']
            salaire = row['revenu_mensuel']

            if poste not in self.quartiles_par_poste_:
                return 2

            seuils = self.quartiles_par_poste_[poste]
            if salaire <= seuils[0]:
                return 1
            elif salaire <= seuils[1]:
                return 2
            elif salaire <= seuils[2]:
                return 3
            else:
                return 4

        X_out['quartile_salaire_par_poste'] = X_out.apply(determiner_quartile, axis=1)

        return X_out