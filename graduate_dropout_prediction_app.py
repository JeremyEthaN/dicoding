import streamlit as st
import pandas as pd
import numpy as np
import joblib
from data_preprocessing import (
    data_preprocessing,
    scaler_Application_mode,
    scaler_Application_order,
    scaler_Course,
    scaler_Previous_qualification_grade,
    scaler_Admission_grade,
    scaler_Tuition_fees_up_to_date,
    scaler_Gender,
    scaler_Scholarship_holder,
    scaler_Age_at_enrollment,
    scaler_Curricular_units_1st_sem_enrolled,
    scaler_Curricular_units_1st_sem_approved,
    scaler_Curricular_units_1st_sem_grade,
    scaler_Curricular_units_2nd_sem_enrolled,
    scaler_Curricular_units_2nd_sem_approved,
    scaler_Curricular_units_2nd_sem_grade
)
from prediction import prediction

col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://raw.githubusercontent.com/JeremyEthaN/dicoding/main/Belajar%20Penerapan%20Data%20Science/itb%20logo.png", width=100)
with col2:
    st.header('Graduate / Dropout Prediction App (Prototype)')

data = pd.DataFrame()

col1, col2, col3 = st.columns(3)

with col1:
    Application_mode = int(st.number_input(label='Application_mode', value=1))
    data["Application_mode"] = scaler_Application_mode.transform(np.asarray(Application_mode).reshape(-1, 1))[0]

with col2:
    Application_order = int(st.number_input(label='Application_order', value=1))
    data["Application_order"] = scaler_Application_order.transform(np.asarray(Application_order).reshape(-1, 1))[0]

with col3:
    Course = int(st.number_input(label='Course', value=1))
    data["Course"] = scaler_Course.transform(np.asarray(Course).reshape(-1, 1))[0]

col1, col2, col3 = st.columns(3)

with col1:
    Previous_qualification_grade = float(st.number_input(label='Previous_qualification_grade', value=12.0))
    data["Previous_qualification_grade"] = scaler_Previous_qualification_grade.transform(np.asarray(Previous_qualification_grade).reshape(-1, 1))[0]

with col2:
    Admission_grade = float(st.number_input(label='Admission_grade', value=13.0))
    data["Admission_grade"] = scaler_Admission_grade.transform(np.asarray(Admission_grade).reshape(-1, 1))[0]

with col3:
    Tuition_fees_up_to_date = int(st.number_input(label='Tuition_fees_up_to_date', value=1))
    data["Tuition_fees_up_to_date"] = scaler_Tuition_fees_up_to_date.transform(np.asarray(Tuition_fees_up_to_date).reshape(-1, 1))[0]

col1, col2, col3 = st.columns(3)

with col1:
    Gender = int(st.number_input(label='Gender', value=1))
    data["Gender"] = scaler_Gender.transform(np.asarray(Gender).reshape(-1, 1))[0]

with col2:
    Scholarship_holder = int(st.number_input(label='Scholarship_holder', value=0))
    data["Scholarship_holder"] = scaler_Scholarship_holder.transform(np.asarray(Scholarship_holder).reshape(-1, 1))[0]

with col3:
    Age_at_enrollment = int(st.number_input(label='Age_at_enrollment', value=18))
    data["Age_at_enrollment"] = scaler_Age_at_enrollment.transform(np.asarray(Age_at_enrollment).reshape(-1, 1))[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    Curricular_units_1st_sem_enrolled = int(st.number_input(label='Curricular_units_1st_sem_enrolled', value=5))
    data["Curricular_units_1st_sem_enrolled"] = scaler_Curricular_units_1st_sem_enrolled.transform(np.asarray(Curricular_units_1st_sem_enrolled).reshape(-1, 1))[0]

with col2:
    Curricular_units_1st_sem_approved = int(st.number_input(label='Curricular_units_1st_sem_approved', value=5))
    data["Curricular_units_1st_sem_approved"] = scaler_Curricular_units_1st_sem_approved.transform(np.asarray(Curricular_units_1st_sem_approved).reshape(-1, 1))[0]

with col3:
    Curricular_units_1st_sem_grade = float(st.number_input(label='Curricular_units_1st_sem_grade', value=14.0))
    data["Curricular_units_1st_sem_grade"] = scaler_Curricular_units_1st_sem_grade.transform(np.asarray(Curricular_units_1st_sem_grade).reshape(-1, 1))[0]

with col4:
    Curricular_units_2nd_sem_enrolled = int(st.number_input(label='Curricular_units_2nd_sem_enrolled', value=5))
    data["Curricular_units_2nd_sem_enrolled"] = scaler_Curricular_units_2nd_sem_enrolled.transform(np.asarray(Curricular_units_2nd_sem_enrolled).reshape(-1, 1))[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    Curricular_units_2nd_sem_approved = int(st.number_input(label='Curricular_units_2nd_sem_approved', value=5))
    data["Curricular_units_2nd_sem_approved"] = scaler_Curricular_units_2nd_sem_approved.transform(np.asarray(Curricular_units_2nd_sem_approved).reshape(-1, 1))[0]

with col2:
    Curricular_units_2nd_sem_grade = float(st.number_input(label='Curricular_units_2nd_sem_grade', value=14.0))
    data["Curricular_units_2nd_sem_grade"] = scaler_Curricular_units_2nd_sem_grade.transform(np.asarray(Curricular_units_2nd_sem_grade).reshape(-1, 1))[0]

with st.expander("View the Raw Data"):
    st.dataframe(data=data, width=800, height=10)

if st.button('Predict'):
    new_data = data_preprocessing(data=data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    st.write("Prediction: {}".format(prediction(new_data)))
