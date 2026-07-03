import datetime
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models import utils
from models.technova_correlation_cleaning import CorrelationFilter
from models.technova_features import TechNovaFeatureEngineering

base_dir = Path(__file__).resolve().parent.parent

@pytest.fixture
def employee_df():
    data_sirh = pd.read_csv(base_dir / 'data/raw/extrait_sirh.csv', sep=',', na_values=[''], quotechar='"')
    data_eval = pd.read_csv(base_dir / 'data/raw/extrait_eval.csv', sep=',', na_values=[''], quotechar='"')
    data_sondage = pd.read_csv(base_dir / 'data/raw/extrait_sondage.csv', sep=',', na_values=[''], quotechar='"')

    data_eval['id_employee'] = data_eval['eval_number'].apply(lambda x : x.split('E_')[1])
    data_eval['id_employee'] = data_eval['id_employee'].astype(int)

    data_sondage['id_employee'] = data_sondage['code_sondage']
    data_sondage['id_employee'] = data_sondage['id_employee'].astype(int)

    data = data_sirh.merge(data_eval, how='inner')
    data = data.merge(data_sondage, how='inner')

    return data

@pytest.fixture
def split_data(employee_df):
    return utils.split_train_data(employee_df, "a_quitte_l_entreprise")

@pytest.fixture
def trained_feature_engineer(split_data):
    fe = TechNovaFeatureEngineering()
    fe.fit(split_data["X_train"])
    return fe

@pytest.fixture
def pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), make_column_selector(dtype_include="number")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include="object")),
        ],
        remainder="drop",
    )
    feature_selector = SelectFromModel(
        estimator=RandomForestClassifier(n_estimators=10, random_state=42),
        max_features=5,
        threshold=-np.inf,
    )
    lr = LogisticRegression(random_state=42, max_iter=500, class_weight="balanced", solver="saga")
    threshold_model = FixedThresholdClassifier(estimator=lr, threshold=0.5, response_method="predict_proba")
    return Pipeline([
        ("features", TechNovaFeatureEngineering()),
        ("corr_cleaning", CorrelationFilter(threshold=0.80)),
        ("preprocessor", preprocessor),
        ("feature_selection", feature_selector),
        ("model", threshold_model),
    ])

class TestSplitTrainData:
    def test_returns_four_keys(self, split_data):
        assert set(split_data.keys()) == {"X_train", "X_test", "y_train", "y_test"}

    def test_80_20_split(self, employee_df, split_data):
        n = len(employee_df)
        assert len(split_data["X_train"]) == pytest.approx(n * 0.8, abs=1)
        assert len(split_data["X_test"]) == pytest.approx(n * 0.2, abs=1)

    def test_target_not_in_features(self, split_data):
        assert "a_quitte_l_entreprise" not in split_data["X_train"].columns
        assert "a_quitte_l_entreprise" not in split_data["X_test"].columns

class TestTechNovaFeatureEngineering:
    def test_transform_adds_expected_columns(self, trained_feature_engineer, split_data):
        expected_new_cols = {
            "ratio_anciennete_total_experence",
            "ratio_poste_actuel_anciennete",
            "score_penibilite_transport",
            "score_bien_etre",
            "score_performance",
            "evolution_performance",
            "evolution_hierarchie_score",
            "annee_par_experience",
            "quartile_salaire_par_poste",
        }
        result = trained_feature_engineer.transform(split_data["X_train"])
        assert expected_new_cols.issubset(result.columns)

class TestCorrelationFilter:
    def test_fit_returns_self(self, split_data):
        fe = TechNovaFeatureEngineering()
        X = fe.fit_transform(split_data["X_train"])
        cf = CorrelationFilter(threshold=0.80)
        assert cf.fit(X) is cf

    def test_threshold_1_drops_nothing_on_random_data(self, split_data):
        fe = TechNovaFeatureEngineering()
        X_fe = fe.fit_transform(split_data["X_train"])
        cf = CorrelationFilter(threshold=1.0)
        cf.fit(X_fe)
        assert cf.columns_to_drop_ == []

class TestFullPipeline:
    def test_pipeline_fit_does_not_raise_errors(self, pipeline, split_data):
        pipeline.fit(split_data["X_train"], split_data["y_train"])

    def test_predict_output_length_matches_input(self, pipeline, split_data):
        pipeline.fit(split_data["X_train"], split_data["y_train"])
        preds = pipeline.predict(split_data["X_test"])
        assert len(preds) == len(split_data["X_test"])

    def test_pipeline_steps_are_named_correctly(self, pipeline):
        step_names = [name for name, _ in pipeline.steps]
        assert step_names == ["features", "corr_cleaning", "preprocessor", "feature_selection", "model"]

class TestArtifactExport:
    def test_joblib_files_are_created(self, pipeline, split_data, tmp_path):
        pipeline.fit(split_data["X_train"], split_data["y_train"])
        preprocessor_path = tmp_path / "classification_preprocessor.joblib"
        model_path = tmp_path / "classification_model.joblib"

        final_preprocessor = Pipeline(pipeline.steps[:-1])
        final_model = pipeline.named_steps["model"]

        joblib.dump(final_preprocessor, preprocessor_path)
        joblib.dump(final_model, model_path)

        assert preprocessor_path.exists()
        assert model_path.exists()

    def test_loaded_artifacts_produce_predictions(self, pipeline, split_data, tmp_path):
        pipeline.fit(split_data["X_train"], split_data["y_train"])
        preprocessor_path = tmp_path / "classification_preprocessor.joblib"
        model_path = tmp_path / "classification_model.joblib"

        joblib.dump(Pipeline(pipeline.steps[:-1]), preprocessor_path)
        joblib.dump(pipeline.named_steps["model"], model_path)

        loaded_preprocessor = joblib.load(preprocessor_path)
        loaded_model = joblib.load(model_path)

        X_transformed = loaded_preprocessor.transform(split_data["X_test"])
        preds = loaded_model.predict(X_transformed)
        assert len(preds) == len(split_data["X_test"])

    def test_metadata_file_has_required_keys(self, tmp_path):
        version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "version": version,
            "n_features_out": 15,
            "target": "a_quitte_l_entreprise",
            "sklearn_version": sklearn.__version__,
        }
        meta_path = tmp_path / f"metadata_{version}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

        with open(meta_path) as f:
            loaded = json.load(f)

        assert {"version", "n_features_out", "target", "sklearn_version"} == loaded.keys()
        assert loaded["target"] == "a_quitte_l_entreprise"
        assert loaded["n_features_out"] == 15
        assert loaded["sklearn_version"] == sklearn.__version__
