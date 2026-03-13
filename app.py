import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the pre-trained model and label encoder
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# 2. Define the Streamlit application title and description
st.title('Asthma Risk Prediction App')
st.write('Enter the patient clinical parameters to predict the asthma risk category.')

# 3. Create input fields for all 13 features
# Helper function to create number input with consistent styling
def number_input_with_label(label, min_value, max_value, value, step):
    return st.number_input(label, min_value=float(min_value), max_value=float(max_value), value=float(value), step=float(step), format="%.2f")

# Input fields for clinical parameters
st.subheader('Clinical Parameters')
pefr = number_input_with_label('PEFR (L/min)', 80, 680, 350, 1.0)
respiratory_rate = number_input_with_label('Respiratory Rate (breaths/min)', 12, 43, 20, 0.1)
heart_rate = number_input_with_label('Heart Rate (bpm)', 60, 145, 90, 0.1)
spo2 = number_input_with_label('SpO₂ (%)', 82, 100, 95, 0.1)
height = number_input_with_label('Height (cm)', 138, 185, 160, 0.1)
absolute_eosinophil_count = st.number_input('Absolute Eosinophil Count (cells/µL)', min_value=50, max_value=2400, value=750, step=1)

# Input fields for derived indices (calculated based on clinical parameters)
st.subheader('Derived Indices')
afr = number_input_with_label('AFR (PEFR/Height)', 0.5, 4.8, 2.0, 0.01)
bsi = number_input_with_label('BSI (RR × HR)', 700, 6000, 2000, 1.0)
oer = number_input_with_label('OER (SpO₂/RR)', 2.0, 8.5, 4.5, 0.01)
ali = number_input_with_label('ALI (AEC/100)', 0.5, 24.0, 7.5, 0.01)

# Input fields for risk scoring
st.subheader('Risk Scoring')
risk_score = number_input_with_label('Risk Score', -8.5, 13.5, 0.0, 0.01)
probability = number_input_with_label('Probability', 0.0, 1.0, 0.5, 0.0001)
probability_percent = number_input_with_label('Probability (%)', 0.0, 100.0, 50.0, 0.01)

# 4. Add a 'Predict' button
if st.button('Predict Asthma Risk'):
    # 5. Gather input values into a pandas DataFrame
    input_data = pd.DataFrame([[pefr, respiratory_rate, heart_rate, spo2, height, absolute_eosinophil_count,
                                  afr, bsi, oer, ali, risk_score, probability, probability_percent]],
                              columns=['PEFR (L/min)', 'Respiratory Rate (breaths/min)', 'Heart Rate (bpm)',
                                       'SpO₂ (%)', 'Height (cm)', 'Absolute Eosinophil Count (cells/µL)',
                                       'AFR (PEFR/Height)', 'BSI (RR × HR)', 'OER (SpO₂/RR)', 'ALI (AEC/100)',
                                       'Risk Score', 'Probability', 'Probability (%)'])

    # 6. Use the loaded model to make a prediction
    prediction_numeric = model.predict(input_data)
    
    # 7. Inverse transform the numerical prediction back to the original 'Risk Category' string
    prediction_category = label_encoder.inverse_transform(prediction_numeric)

    # 8. Display the predicted 'Risk Category' to the user
    st.success(f'Predicted Asthma Risk Category: {prediction_category[0]}')
