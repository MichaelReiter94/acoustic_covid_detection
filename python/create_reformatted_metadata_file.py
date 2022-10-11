import pandas as pd

meta_data_labels = {
    "id": "user_id",
    "a": "age",
    "covid_status": "covid_health_status",
    "record_date": "record_date",
    "ep": "english_proficiency",
    "g": "gender",
    "l_c": "country",
    "l_l": "local_region",
    "l_s": "state",
    "rU": "returning_user",
    "asthma": "asthma",
    "cough": "cough",
    "smoker": "smoker",
    "test_status": "covid_test_result",
    "ht": "hypertension",
    "cold": "cold",
    "diabetes": "diabetes",
    "diarrhoea": "diarrheoa",
    "um": "was_using_mask",
    "ihd": "ischemic_heart_disease",
    "bd": "breathing_difficulties",
    "st": "sore_throat",
    "fever": "fever",
    "ftg": "fatigue",
    "mp": "muscle_pain",
    "loss_of_smell": "loss_of_smell",
    "cld": "chronic_lung_disease",
    "pneumonia": "pneumonia",
    "ctScan": "has_taken_ct_scan",
    "testType": "type_of_covid_test",
    "test_date": "covid_test_date",
    "vacc": "vaccination_status",  # (y->both doses, p->one dose(partially vaccinated), n->no doses)
    "ctDate": "date_of_ct_scan",
    "ctScore": "ct_score",
    "others_resp": "other_respiratory_illness",
    "others_preexist": "other_preexisting_condition"
}

metadata = pd.read_csv("data/Coswara_processed/combined_data.csv")
# metadata.info()
metadata.rename(meta_data_labels, axis="columns", inplace=True)
# metadata.info()

metadata.to_csv("data/Coswara_processed/reformatted_metadata.csv")
