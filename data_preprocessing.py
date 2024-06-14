import joblib
import numpy as np
import pandas as pd
scaler_Application_mode = joblib.load("model/scaler_Application_mode.joblib")
scaler_Application_order = joblib.load("model/scaler_Application_order.joblib")
scaler_Course = joblib.load("model/scaler_Course.joblib")
scaler_Previous_qualification_grade = joblib.load("model/scaler_Previous_qualification_grade.joblib")
scaler_Admission_grade = joblib.load("model/scaler_Admission_grade.joblib")
scaler_Tuition_fees_up_to_date = joblib.load("model/scaler_Tuition_fees_up_to_date.joblib")
scaler_Gender = joblib.load("model/scaler_Gender.joblib")
scaler_Scholarship_holder = joblib.load("model/scaler_Scholarship_holder.joblib")
scaler_Age_at_enrollment = joblib.load("model/scaler_Age_at_enrollment.joblib")
scaler_Curricular_units_1st_sem_enrolled = joblib.load("model/scaler_Curricular_units_1st_sem_enrolled.joblib")
scaler_Curricular_units_1st_sem_approved = joblib.load("model/scaler_Curricular_units_1st_sem_approved.joblib")
scaler_Curricular_units_1st_sem_grade = joblib.load("model/scaler_Curricular_units_1st_sem_grade.joblib")
scaler_Curricular_units_2nd_sem_enrolled = joblib.load("model/scaler_Curricular_units_2nd_sem_enrolled.joblib")
scaler_Curricular_units_2nd_sem_approved = joblib.load("model/scaler_Curricular_units_2nd_sem_approved.joblib")
scaler_Curricular_units_2nd_sem_grade = joblib.load("model/scaler_Curricular_units_2nd_sem_grade.joblib")
def data_preprocessing(data):
    """Preprocessing data
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the data to make prediction 
        
    return:
        Pandas DataFrame: Dataframe that contain all the preprocessed data
    """
    data = data.copy()
    df = pd.DataFrame()

    df["Application_mode"] = scaler_Application_mode.transform(np.asarray(data["Application_mode"]).reshape(-1, 1)).flatten()
    df["Application_order"] = scaler_Application_order.transform(np.asarray(data["Application_order"]).reshape(-1, 1)).flatten()
    df["Course"] = scaler_Course.transform(np.asarray(data["Course"]).reshape(-1, 1)).flatten()
    df["Previous_qualification_grade"] = scaler_Previous_qualification_grade.transform(np.asarray(data["Previous_qualification_grade"]).reshape(-1, 1)).flatten()
    df["Admission_grade"] = scaler_Admission_grade.transform(np.asarray(data["Admission_grade"]).reshape(-1, 1)).flatten()
    df["Tuition_fees_up_to_date"] = scaler_Tuition_fees_up_to_date.transform(np.asarray(data["Tuition_fees_up_to_date"]).reshape(-1, 1)).flatten()
    df["Gender"] = scaler_Gender.transform(np.asarray(data["Gender"]).reshape(-1, 1)).flatten()
    df["Scholarship_holder"] = scaler_Scholarship_holder.transform(np.asarray(data["Scholarship_holder"]).reshape(-1, 1)).flatten()
    df["Age_at_enrollment"] = scaler_Age_at_enrollment.transform(np.asarray(data["Age_at_enrollment"]).reshape(-1, 1)).flatten()
    df["Curricular_units_1st_sem_enrolled"] = scaler_Curricular_units_1st_sem_enrolled.transform(np.asarray(data["Curricular_units_1st_sem_enrolled"]).reshape(-1, 1)).flatten()
    df["Curricular_units_1st_sem_approved"] = scaler_Curricular_units_1st_sem_approved.transform(np.asarray(data["Curricular_units_1st_sem_approved"]).reshape(-1, 1)).flatten()
    df["Curricular_units_1st_sem_grade"] = scaler_Curricular_units_1st_sem_grade.transform(np.asarray(data["Curricular_units_1st_sem_grade"]).reshape(-1, 1)).flatten()
    df["Curricular_units_2nd_sem_enrolled"] = scaler_Curricular_units_2nd_sem_enrolled.transform(np.asarray(data["Curricular_units_2nd_sem_enrolled"]).reshape(-1, 1)).flatten()
    df["Curricular_units_2nd_sem_approved"] = scaler_Curricular_units_2nd_sem_approved.transform(np.asarray(data["Curricular_units_2nd_sem_approved"]).reshape(-1, 1)).flatten()
    df["Curricular_units_2nd_sem_grade"] = scaler_Curricular_units_2nd_sem_grade.transform(np.asarray(data["Curricular_units_2nd_sem_grade"]).reshape(-1, 1)).flatten()

    return df
