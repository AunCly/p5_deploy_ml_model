# %% [markdown]
# # Cycle 7 : Approche finale
# 
# %%
### Import des modules
# %load_ext autoreload
# %autoreload 2

import joblib
import pandas as pd
import numpy as np
from models import utils
from models.technova_features import TechNovaFeatureEngineering
from models.technova_correlation_cleaning import CorrelationFilter
import sklearn
from sklearn.model_selection import FixedThresholdClassifier, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, f1_score, roc_auc_score, classification_report, make_scorer, fbeta_score
from scipy.stats import uniform, loguniform
import json, datetime

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# %%
data = pd.read_csv('../data/rafined/employees.csv', sep=',')
# %%
train_data = utils.split_train_data(data, 'a_quitte_l_entreprise')
# %%
# Encodage des variables catégorielles et standardisation des variables numériques
numeric_features = data.select_dtypes(include=['int64', 'float64'])
categorical_features = ['statut_marital', 'departement', 'poste', 'domaine_etude']

preprocessor = ColumnTransformer(
    transformers=[
        ('standard_scaler', StandardScaler(), make_column_selector(dtype_include='number')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_include='object')),
    ],
    remainder='drop'
)
# %%
nb_features_to_keep = 15
# %%
# Permet de ne garder que nb_feature_to_keep pour réduire le bruit
feature_selector = SelectFromModel(
    estimator=RandomForestClassifier(n_estimators=50, random_state=42),
    max_features=nb_features_to_keep,
    threshold=-np.inf
)
# %%
# Benchmark du modele avec CV
logistic_regression_model = LogisticRegression(
    random_state=42,
    max_iter=5000,
    class_weight='balanced',
    solver='saga'
)

threshold_model = FixedThresholdClassifier(
    estimator=logistic_regression_model,
    threshold=0.5,
    response_method="predict_proba"
)

pipeline = Pipeline([
    ('features', TechNovaFeatureEngineering()),
    ('corr_cleaning', CorrelationFilter(threshold=0.80)),
    ('preprocessor', preprocessor),
    ('feature_selection', feature_selector),
    ('model', threshold_model),
])

utils.benchmark(pipeline, train_data)
# %%
# Recherche des meilleurs hyper-paramètres
train_data = utils.split_train_data(data, 'a_quitte_l_entreprise')

full_pipeline = Pipeline([
    ('features', TechNovaFeatureEngineering()),
    ('corr_cleaning', CorrelationFilter(threshold=0.80)),
    ('preprocessor', preprocessor),
    ('feature_selection', feature_selector),
    ('model', threshold_model)
])

param_distributions = {
    'model__estimator__C': loguniform(1e-4, 1e2),
    'model__estimator__l1_ratio': uniform(0, 1),
    'model__threshold': [0.3, 0.4, 0.5, 0.6],
}

f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

search = RandomizedSearchCV(
    estimator=full_pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    scoring=f2_scorer,
    cv=5,
    random_state=42,
    n_jobs=-1,
)

search.fit(train_data['X_train'], train_data['y_train'])

print(f"Meilleurs paramètres : {search.best_params_}")
print(f"f1 moyen en Validation Croisée : {search.best_score_:.4f}\n")

best_pipeline = search.best_estimator_
# %%
# Affichage des performances du modèle
utils.benchmark(best_pipeline, train_data)

y_pred_test = best_pipeline.predict(train_data['X_test'])
y_probs_test = best_pipeline.predict_proba(train_data['X_test'])[:, 1]
y_pred_test = best_pipeline.predict(train_data['X_test'])

y_probs_train = best_pipeline.predict_proba(train_data['X_train'])[:, 1]
y_pred_train = best_pipeline.predict(train_data['X_train'])

auc_train = roc_auc_score(train_data['y_train'], y_probs_train)
auc_test = roc_auc_score(train_data['y_test'], y_probs_test)

recall_train = recall_score(train_data['y_train'], y_pred_train)
recall_test = recall_score(train_data['y_test'], y_pred_test)

f1_train = f1_score(train_data['y_train'], y_pred_train)
f1_test = f1_score(train_data['y_test'], y_pred_test)

print("Vérif overfit")
print(f"ROC AUC - Train Set : {auc_train:.4f}")
print(f"ROC AUC - Test Set  : {auc_test:.4f}")
print(f"Différence ROC AUC (Overfit) : {auc_train - auc_test:.4f}\n")
print(f"Recall - Train Set : {recall_train:.4f}")
print(f"Recall - Test Set  : {recall_test:.4f}")
print(f"Différence Recall (Overfit) : {recall_train - recall_test:.4f}\n")
print(f"f1 - Train Set : {f1_train:.4f}")
print(f"f1 - Test Set  : {f1_test:.4f}")
print(f"Différence f1 (Overfit) : {f1_train - f1_test:.4f}\n")

print("Performances")
print(classification_report(train_data['y_test'], y_pred_test))

print("Matrice de Confusion :")
print(confusion_matrix(train_data['y_test'], y_pred_test))
# %%
# Entrainement final avec 100% des données + les parametres trouvé par RandomizedSearchCV
best_params = search.best_params_

X_complet = data.drop('a_quitte_l_entreprise', axis=1)
y_complet = data['a_quitte_l_entreprise']

production_pipeline = Pipeline([
    ('features', TechNovaFeatureEngineering()),
    ('corr_cleaning', CorrelationFilter(threshold=0.80)),
    ('preprocessor', preprocessor),
    ('feature_selection', feature_selector),
    ('model', FixedThresholdClassifier(
        estimator=LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=5000,
            solver='saga'
        ),
        threshold=0.5,
        response_method="predict_proba"
    ))
])

production_pipeline.set_params(**best_params)

production_pipeline.fit(X_complet, y_complet)
# %%
#Export des modèles pour la production...

from pathlib import Path
import os

base_dir = Path(os.getcwd()).resolve().parent

if not os.path.exists(base_dir / 'models/compiled/'):
    os.mkdir(base_dir / 'models/compiled/')

version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

metadata = {
    "version": version,
    "n_features_out": nb_features_to_keep,
    "target": "a_quitte_l_entreprise",
    "sklearn_version": sklearn.__version__,
}

final_preprocessor = Pipeline(production_pipeline.steps[:-1])
final_model = production_pipeline.named_steps['model']

joblib.dump(final_preprocessor, base_dir / 'models/compiled/classification_preprocessor.joblib')
joblib.dump(final_model, base_dir / 'models/compiled/classification_model.joblib')

with open(base_dir / f'models/compiled/metadata_{version}.json', 'w') as f:
    json.dump(metadata, f, indent=2)