from pathlib import Path
import pandas as pd
import database.database as database

def seed():

    base_dir = Path(__file__).resolve().parent.parent

    data_sirh = pd.read_csv( base_dir / 'data/raw/extrait_sirh.csv', sep=',', na_values=[''], quotechar='"')
    data_eval = pd.read_csv( base_dir / 'data/raw/extrait_eval.csv', sep=',', na_values=[''], quotechar='"')
    data_sondage = pd.read_csv( base_dir / 'data/raw/extrait_sondage.csv', sep=',', na_values=[''], quotechar='"')
    engine = database.get_engine()

    try:

        data_sirh['id'] = data_sirh['id_employee']
        data_sirh['id'] = data_sirh['id'].astype(int)
        data_sirh.drop(columns=['id_employee'], inplace=True)

        data_sirh.to_sql(
            "employees",
            engine,
            if_exists="append",
            index=False,
            chunksize=1000,
            method="multi",
        )

        data_eval['employee_id'] = data_eval['eval_number'].apply(lambda x: x.split('E_')[1])
        data_eval['employee_id'] = data_eval['employee_id'].astype(int)

        data_eval.to_sql(
            "evaluations",
            engine,
            if_exists="append",
            index=False,
            chunksize=1000,
            method="multi",
        )

        data_sondage['employee_id'] = data_sondage['code_sondage']
        data_sondage['employee_id'] = data_sondage['employee_id'].astype(int)

        data_sondage.to_sql(
            "sondages",
            engine,
            if_exists="append",
            index=False,
            chunksize=1000,
            method="multi",
        )
    except Exception as e:
        print(e)
