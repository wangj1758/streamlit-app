import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Prediction model for ICU admission in AECOPD patients",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        background-color: #4e8df5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    h1, h2, h3 {
        color: #1e3d8f;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained XGBoost model
model = joblib.load('XGBoost.pkl')

# Function to predict ICU admission using the loaded XGBoost model
def predict_icu_admission(features):
    # Use the trained XGBoost model to make predictions
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)
    return prediction[0], probabilities

# Function to calculate feature importance using SHAP
def get_feature_importance(model, features_df, feature_names):
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(features_df)
        
        # For binary classification, take the SHAP values for class 1 (positive class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP values for each feature
        importance = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame with feature names and importance
        df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        return df.sort_values('Importance', ascending=False), explainer, shap_values
    except Exception as e:
        st.error(f"Error calculating SHAP values: {e}")
        return pd.DataFrame({'Feature': feature_names, 'Importance': np.zeros(len(feature_names))}), None, None

# Function to create SHAP force plot and return the image as base64 encoded string
def get_shap_force_plot(explainer, shap_values, features_df, prediction_class):
    try:
        # Create a matplotlib figure
        plt.figure(figsize=(12, 4))
        
        # If binary classification, we use the appropriate class's SHAP values
        if isinstance(shap_values, list):
            selected_shap_values = shap_values[1]  # Use class 1 (positive class) for binary classification
        else:
            selected_shap_values = shap_values
        
        # Create the SHAP force plot
        shap.force_plot(
            explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
            selected_shap_values, 
            features_df,
            matplotlib=True,
            show=False,
            feature_names=features_df.columns.tolist()
        )
        
        # Add title based on prediction
        plt.title(f"SHAP Force Plot - {'High Risk' if prediction_class == 1 else 'Low Risk'} for ICU Admission", 
                fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert plot to base64 encoded image
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64
    except Exception as e:
        st.error(f"Error creating SHAP force plot: {e}")
        return None

# Title and introduction
st.title("Prediction of ICU Admission in Acute Exacerbation of COPD")
st.markdown("""
This application implements a machine learning model for predicting ICU admission 
in patients with Acute Exacerbation of Chronic Obstructive Pulmonary Disease (AECOPD).
""")

# Create sidebar for input parameters
st.sidebar.header("Patient Information")

# Demographics section
with st.sidebar.expander("Demographics", expanded=True):
    age = st.number_input("Age (years)", min_value=18, max_value=110, value=65)
    
    gender = st.radio(
        "Gender",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )
    
    race_options = {
        1: "White",
        2: "Black",
        3: "Asian",
        4: "Hispanic or Latino",
        5: "Other",
        6: "Unknown"
    }
    race = st.selectbox(
        "Race",
        options=list(race_options.keys()),
        format_func=lambda x: race_options[x]
    )
    
    marital_options = {
        0: "Single",
        1: "Married",
        2: "Divorced",
        3: "Widowed"
    }
    marital = st.selectbox(
        "Marital Status",
        options=list(marital_options.keys()),
        format_func=lambda x: marital_options[x]
    )

# Comorbidities section
with st.sidebar.expander("Comorbidities", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        pna = st.radio(
            "Pneumonia",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        
        hf = st.radio(
            "Heart Failure",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
    
    with col2:
        ckd = st.radio(
            "Chronic Kidney Disease",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        
        aki = st.radio(
            "Acute Kidney Injury",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

# Laboratory Values section
with st.sidebar.expander("Laboratory Values (Blood)", expanded=True):
    # Metabolic parameters
    st.subheader("Metabolic Parameters")
    glu = st.number_input("Glucose (mg/dL)", min_value=40, max_value=600, value=120)
    bun = st.number_input("Urea Nitrogen (mg/dL)", min_value=5, max_value=200, value=20)
    
    # Electrolytes
    st.subheader("Electrolytes")
    k = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=8.0, value=4.0, step=0.1)
    mg = st.number_input("Magnesium (mg/dL)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    cl = st.number_input("Chloride (mEq/L)", min_value=80, max_value=130, value=100)
    ag = st.number_input("Anion Gap (mEq/L)", min_value=3, max_value=30, value=12)
    tc = st.number_input("Total Calcium (mg/dL)", min_value=5.0, max_value=15.0, value=9.0, step=0.1)
    
    # Hematology parameters
    st.subheader("Hematology Parameters")
    wbc = st.number_input("White Blood Cells (K/uL)", min_value=0.5, max_value=50.0, value=9.0, step=0.1)
    rbc = st.number_input("Red Blood Cells (m/uL)", min_value=2.0, max_value=7.0, value=4.5, step=0.1)
    rdw = st.number_input("Red Cell Distribution Width (%)", min_value=10.0, max_value=30.0, value=14.0, step=0.1)
    mchc = st.number_input("Mean Corpuscular Hemoglobin Concentration (g/dL)", min_value=25.0, max_value=38.0, value=33.0, step=0.1)
    plt = st.number_input("Platelet Count (K/uL)", min_value=20, max_value=1000, value=250)

# Button to make prediction
predict_button = st.button("Predict ICU Admission")

if predict_button:
    # Collect features in the EXACT same order as used in training
    selected_features = ['WBC', 'PLT', 'GLU', 'RBC', 'Cl', 'BUN', 'TC', 'AG', 'RDW', 'K', 
                         'Age', 'Mg', 'MCHC', 'Race', 'AKI', 'PNA', 'CKD', 'Marital status', 
                         'Gender', 'HF']
    
    # Create a new DataFrame with features in the correct order
    features_df = pd.DataFrame([[
        wbc, plt, glu, rbc, cl, bun, tc, ag, rdw, k, 
        age, mg, mchc, race, aki, pna, ckd, marital, 
        gender, hf
    ]], columns=selected_features)
    
    # Make prediction
    prediction, probabilities = predict_icu_admission(features_df)
    probability_icu = probabilities[0][1] * 100
    
    # Get feature importance and SHAP explainer/values
    importance_result = get_feature_importance(model, features_df, selected_features)
    
    if len(importance_result) == 3:
        importance_df, explainer, shap_values = importance_result
    else:
        importance_df = importance_result
        explainer, shap_values = None, None
    
    # Create columns for results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Prediction Results")
        
        # Display prediction with appropriate formatting
        if prediction == 1:
            st.markdown(f"""
            <div class="risk-high">
                <h3>High Risk for ICU Admission</h3>
                <p>Probability: {probability_icu:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <b>Clinical Interpretation:</b> This patient shows a high risk profile for requiring 
            ICU admission during AECOPD hospitalization. Consider:
            <ul>
                <li>Early pulmonology consultation</li>
                <li>Close monitoring for respiratory failure</li>
                <li>Early assessment of ventilatory support needs</li>
                <li>Consider ICU bed availability</li>
            </ul>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <h3>Low Risk for ICU Admission</h3>
                <p>Probability: {probability_icu:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <b>Clinical Interpretation:</b> This patient shows a lower risk profile for requiring 
            ICU admission. Consider:
            <ul>
                <li>Standard AECOPD management protocol</li>
                <li>Regular monitoring as per institutional guidelines</li>
                <li>Re-assessment if clinical status changes</li>
            </ul>
            """, unsafe_allow_html=True)
    
    with col2:
        st.header("Feature Importance (SHAP)")
        
        # Display feature importance bar chart
        st.bar_chart(importance_df.set_index('Feature')['Importance'])
        
        st.markdown("""
        This chart shows the relative importance of each feature in the prediction model
        calculated using SHAP (SHapley Additive exPlanations) values. SHAP values provide a unified
        measure of feature importance based on game theory principles and represent how much
        each feature contributes to the prediction.
        """)
    
    # SHAP Force Plot section
    st.header("SHAP Force Plot")
    
    if explainer is not None and shap_values is not None:
        # Create and display SHAP force plot
        force_plot_image = get_shap_force_plot(explainer, shap_values, features_df, prediction)
        
        if force_plot_image:
            st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{force_plot_image}" style="max-width: 100%;">
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            The SHAP force plot shows how each feature contributes to pushing the model output from 
            the base value (average model output) to the final prediction. Red features push the prediction 
            higher (toward ICU admission), while blue features push the prediction lower (away from ICU admission).
            The size of each feature's bar represents the magnitude of its impact on this specific prediction.
            """)
        else:
            st.warning("Unable to generate SHAP force plot. This may be due to an error in SHAP calculation.")
    else:
        st.warning("SHAP explainer could not be created. Feature importance is still calculated but the force plot is unavailable.")
    
    # Clinical interpretation section
    st.header("Clinical Interpretation Guide")
    
    st.markdown("""
    ### Key Risk Factors in AECOPD
    
    The prediction model identifies several important risk factors that influence ICU admission likelihood:
    
    1. **Laboratory Abnormalities**: Significant deviations in key parameters (Total Calcium, White Blood Cells, Glucose)
    2. **Comorbidities**: Presence of Acute Kidney Injury, Pneumonia, or Chronic Kidney Disease
    3. **Demographic Factors**: Advanced age and certain demographic characteristics
    
    ### Limitations
    
    - This prediction model should be used as a decision support tool, not as a replacement for clinical judgment
    - The model does not account for treatment interventions after admission
    - Local institutional protocols for ICU admission criteria may vary
    """)

else:
    # Display information before prediction
    st.info("Enter patient information in the sidebar and click 'Predict ICU Admission' to generate a prediction.")
    
    # Display model information
    st.header("About the Model")
    
    st.markdown("""
    ### Model Overview
    
    This machine learning model was developed to predict the need for ICU admission in patients
    presenting with Acute Exacerbation of Chronic Obstructive Pulmonary Disease (AECOPD).
    
    **Key Features:**
    
    - The model utilizes 20 easily available clinical variables
    - Includes demographic information, comorbidities, and laboratory values
    - Provides probability estimates and feature importance for interpretability
    - Uses SHAP values to explain individual predictions
    
    ### Variables Used
    
    **Demographics**:
    - Age (Years)
    - Gender (0=Female, 1=Male)
    - Race (1=White, 2=Black, 3=Asian, 4=Hispanic or Latino, 5=Other, 6=Unknown)
    - Marital status (0=Single, 1=Married, 2=Divorced, 3=Widowed)
    
    **Comorbidities**:
    - PNA: Pneumonia (0=No, 1=Yes)
    - HF: Heart failure (0=No, 1=Yes)
    - CKD: Chronic kidney disease (0=No, 1=Yes)
    - AKI: Acute kidney injury (0=No, 1=Yes)
    
    **Laboratory Values (Blood)**:
    - GLU: Glucose (mg/dL)
    - WBC: White Blood Cells (K/uL)
    - RBC: Red Blood Cells (m/uL)
    - RDW: Red Cell Distribution Width (%)
    - MCHC: Mean Corpuscular Hemoglobin Concentration (g/dL)
    - PLT: Platelet Count (K/uL)
    - BUN: Urea Nitrogen (mg/dL)
    - K: Potassium (mEq/L)
    - Mg: Magnesium (mg/dL)
    - Cl: Chloride (mEq/L)
    - AG: Anion Gap (mEq/L)
    - TC: Total Calcium (mg/dL)
    """)
    
    # Model performance information - Update with your actual metrics
    st.header("Model Performance")
    
    st.markdown("""
    ### Validation Metrics
    
    - **AUROC**: 0.773 (95% CI: 0.748-0.799)
    - **Sensitivity**: 0.75
    - **Specificity**: 0.72
    - **PPV**: 0.69
    - **NPV**: 0.78
    - **Accuracy**: 0.73
    
    The model demonstrates good discriminative ability with an AUROC of 0.773, 
    indicating effective classification performance for ICU admission prediction in AECOPD patients.
    """)

# Footer
st.markdown("""
---
<div style="text-align: center; color: gray; font-size: 0.8em;">
This application is intended for research and educational purposes only. Clinical decisions should 
be made based on comprehensive patient evaluation by qualified healthcare professionals.
</div>
""", unsafe_allow_html=True)