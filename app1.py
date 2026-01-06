import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from openai import OpenAI
import os
import base64


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


logo_base64 = get_base64_image("logo.png")

# Page configuration
st.set_page_config(page_title="Stroke Mortality Risk Prediction",
                   page_icon="🏥",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .risk-low {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .risk-moderate {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
""",
            unsafe_allow_html=True)


# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client with API key"""
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get(
        "OPENAI_API_KEY", "")
    if not api_key:
        st.warning(
            "⚠️ OpenAI API key not found. Please set OPENAI_API_KEY environment variable or add to Streamlit secrets."
        )
        return None
    return OpenAI(api_key=api_key)


# Load models
@st.cache_resource
def load_models():
    """Load all trained models and feature names"""
    try:
        model_bundle = joblib.load('stroke_mortality_model_bundle.pkl')
        return model_bundle
    except FileNotFoundError:
        st.error(
            "❌ Model files not found. Please ensure 'stroke_mortality_model_bundle.pkl' is in the same directory."
        )
        st.stop()


# Load SHAP explainers
@st.cache_resource
def load_shap_explainers(_models):
    """Load SHAP explainers for each model"""
    explainers = {}
    for horizon, model in _models.items():
        explainers[horizon] = shap.TreeExplainer(model)
    return explainers


def get_risk_category(risk):
    """Categorize risk level"""
    if risk < 0.15:
        return "Low", "risk-low"
    elif risk < 0.30:
        return "Moderate", "risk-moderate"
    else:
        return "High", "risk-high"


def create_feature_dict(form_data, all_features):
    """Create feature vector from form data"""
    # Initialize all features to 0
    features = {feat: 0 for feat in all_features}

    # Map form data to features
    # Age group
    features['age_group'] = form_data['age_group']

    # Sex
    if form_data['sex'] == 'Male':
        features['Sex_M'] = 1
        features['Sex_F'] = 0
    else:
        features['Sex_M'] = 0
        features['Sex_F'] = 1

    # Race
    if form_data['race'] == 'White':
        features['race_WHITE'] = 1
        features['race_NON-WHITE'] = 0
    else:
        features['race_WHITE'] = 0
        features['race_NON-WHITE'] = 1

    # Blood pressure
    features['blood_pressure'] = 1 if form_data[
        'blood_pressure'] == 'Abnormal' else 0

    # Stroke class
    stroke_classes = ['HEMORRHAGIC', 'ISCHEMIC', 'TIA']
    for sc in stroke_classes:
        features[f'sc_{sc}'] = 1 if form_data['stroke_class'] == sc else 0

    # Marital status
    marital_statuses = ['DIVORCED', 'MARRIED', 'SINGLE', 'WIDOWED']
    for ms in marital_statuses:
        features[
            f'care_level_{ms}'] = 1 if form_data['marital_status'] == ms else 0

    # Readmissions
    features['readmissions'] = 1 if form_data['readmissions'] > 0 else 0

    # Clinical entities (simplified - set to average values from training)
    entity_features = {
        'pct_DOSAGE': form_data.get('pct_DOSAGE', 0),
        'pct_MEDICATION': form_data.get('pct_MEDICATION', 0),
        'pct_SIGN_SYMPTOM': form_data.get('pct_SIGN_SYMPTOM', 0.15),
        'pct_DISEASE_DISORDER': form_data.get('pct_DISEASE_DISORDER', 0.10),
        'pct_BIOLOGICAL_STRUCTURE': form_data.get('pct_BIOLOGICAL_STRUCTURE',
                                                  0.05),
        'pct_LAB_VALUE': form_data.get('pct_LAB_VALUE', 0),
        'pct_DIAGNOSTIC_PROCEDURE': form_data.get('pct_DIAGNOSTIC_PROCEDURE',
                                                  0.05),
        'pct_SEVERITY': form_data.get('pct_SEVERITY', 0.10),
        'pct_THERAPEUTIC_PROCEDURE': form_data.get('pct_THERAPEUTIC_PROCEDURE',
                                                   0),
        'pct_CLINICAL_EVENT': form_data.get('pct_CLINICAL_EVENT', 0),
        'pct_DETAILED_DESCRIPTION': form_data.get('pct_DETAILED_DESCRIPTION',
                                                  0),
        'pct_DURATION': form_data.get('pct_DURATION', 0)
    }
    features.update(entity_features)

    return features


def predict_risks(patient_features, models, feature_names):
    """Predict mortality risks at all time horizons"""
    # Create DataFrame with correct feature order
    patient_df = pd.DataFrame([patient_features])[feature_names]

    risks = {}
    for horizon, model in models.items():
        prob = model.predict_proba(patient_df)[0, 1]
        risks[horizon] = prob

    return risks


def compute_shap_values(patient_features, models, explainers, feature_names):
    """Compute SHAP values for all models"""
    patient_df = pd.DataFrame([patient_features])[feature_names]

    shap_values_dict = {}
    for horizon in models.keys():
        shap_vals = explainers[horizon].shap_values(patient_df)

        # Handle binary classification output
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # Positive class

        shap_values_dict[horizon] = shap_vals[0]

    return shap_values_dict


def get_top_risk_factors(shap_values_dict, feature_names, top_n=5):
    """Get top risk factors across all time horizons"""
    # Average absolute SHAP values across all horizons
    all_shap = np.array(
        [shap_values_dict[h] for h in sorted(shap_values_dict.keys())])
    avg_abs_shap = np.abs(all_shap).mean(axis=0)

    # Get top features
    top_indices = np.argsort(avg_abs_shap)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_values = avg_abs_shap[top_indices]

    return top_features, top_values


def generate_llm_explanation(patient_data, risk, horizon, top_factors,
                             shap_values, feature_names):
    """Generate explanation using GPT-4"""
    client = get_openai_client()

    if client is None:
        return "⚠️ OpenAI API key not configured. Cannot generate AI explanation."

    # Get risk category
    risk_category, _ = get_risk_category(risk)

    # Format top contributing factors
    top_factors_text = []
    for i, (feat, shap_val) in enumerate(zip(top_factors[:5],
                                             shap_values[:5])):
        direction = "increases" if shap_val > 0 else "decreases"
        patient_value = patient_data.get(feat, "N/A")
        top_factors_text.append(
            f"{i+1}. {feat} = {patient_value} ({direction} risk by {abs(shap_val):.3f})"
        )

    factors_str = "\n".join(top_factors_text)

    # Create prompt
    prompt = f"""You are a clinical decision support AI assistant. Explain why this stroke patient has a {risk_category.lower()} risk ({risk:.1%}) of mortality within {horizon} days.

Patient Information:
- Age Group: {patient_data.get('age_group', 'N/A')} (1=18-44, 2=45-64, 3=65-74, 4=75-84, 5=85+)
- Sex: {'Male' if patient_data.get('Sex_M', 0) == 1 else 'Female'}
- Stroke Type: {patient_data.get('stroke_class', 'N/A')}
- Blood Pressure: {'Abnormal' if patient_data.get('blood_pressure', 0) == 1 else 'Normal'}
- Previous Readmissions: {'Yes' if patient_data.get('readmissions', 0) == 1 else 'No'}

Top Contributing Risk Factors (from AI model):
{factors_str}

Write ONE paragraph (4-6 sentences) explaining:
1. Why the patient's risk is {risk_category.lower()} at {horizon} days
2. Which specific factors are most important
3. Brief clinical context for why these factors matter

Use clear, professional language suitable for clinicians. Do not use bullet points."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role":
                "system",
                "content":
                "You are an expert clinical AI assistant specializing in stroke care and mortality risk assessment."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=300)

        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Error generating explanation: {str(e)}"


def plot_risk_trajectory(risks):
    """Plot mortality risk across time horizons"""
    horizons = sorted(risks.keys())
    risk_values = [risks[h] for h in horizons]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot line
    ax.plot(horizons,
            risk_values,
            marker='o',
            linewidth=3,
            markersize=12,
            color='#1f77b4',
            label='Mortality Risk')

    # Color regions
    ax.axhspan(0, 0.15, alpha=0.2, color='green', label='Low Risk')
    ax.axhspan(0.15, 0.30, alpha=0.2, color='yellow', label='Moderate Risk')
    ax.axhspan(0.30, 1.0, alpha=0.2, color='red', label='High Risk')

    # Formatting
    ax.set_xlabel('Time Horizon (days)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mortality Risk', fontsize=14, fontweight='bold')
    ax.set_title('Patient Mortality Risk Trajectory',
                 fontsize=16,
                 fontweight='bold')
    ax.set_ylim([0, 1])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left')

    # Add value labels
    for h, r in zip(horizons, risk_values):
        ax.annotate(f'{r:.1%}',
                    xy=(h, r),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontweight='bold')

    plt.tight_layout()
    return fig


def plot_top_risk_factors(top_features, top_values, patient_features):
    """Plot top risk factors"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create color map based on positive/negative impact
    colors = ['#d62728' if v > 0 else '#2ca02c' for v in top_values]

    # Horizontal bar chart
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_values, color=colors, alpha=0.7)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Average Impact on Risk (SHAP value)',
                  fontsize=12,
                  fontweight='bold')
    ax.set_title('Top 5 Risk Factors Across All Time Horizons',
                 fontsize=14,
                 fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (feat, val) in enumerate(zip(top_features, top_values)):
        patient_val = patient_features.get(feat, "N/A")
        label = f'{val:.3f} (value: {patient_val})'
        ax.text(val, i, f'  {label}', va='center', fontweight='bold')

    plt.tight_layout()
    return fig


# Main app
def main():
    # Initialize session state
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'risks' not in st.session_state:
        st.session_state.risks = None
    if 'patient_features' not in st.session_state:
        st.session_state.patient_features = None
    if 'form_data' not in st.session_state:
        st.session_state.form_data = None
    if 'shap_values_dict' not in st.session_state:
        st.session_state.shap_values_dict = None
    if 'top_features' not in st.session_state:
        st.session_state.top_features = None
    if 'top_values' not in st.session_state:
        st.session_state.top_values = None

    # Header
    st.markdown("""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{logo_base64}" width="120">
            <div class="main-header">Braining's Stroke Mortality Risk Prediction App</div>
        </div>
        """,
                unsafe_allow_html=True)

    st.markdown("""
    This clinical decision support system predicts stroke mortality risk at multiple time horizons
    using machine learning and provides AI-powered explanations of risk factors.
    """)

    # Load models
    model_bundle = load_models()
    models = model_bundle['models']
    feature_names = model_bundle['feature_names']
    horizons = model_bundle['horizons']

    # Load SHAP explainers
    explainers = load_shap_explainers(models)

    st.sidebar.header("📋 Patient Information")

    # Sidebar form for patient input
    with st.sidebar.form("patient_form"):
        st.subheader("Demographics")

        age_group = st.selectbox("Age Group",
                                 options=[1, 2, 3, 4, 5],
                                 format_func=lambda x: {
                                     1: "18-44",
                                     2: "45-64",
                                     3: "65-74",
                                     4: "75-84",
                                     5: "85+"
                                 }.get(x),
                                 index=2)

        sex = st.selectbox("Sex", options=["Male", "Female"])

        race = st.selectbox("Race", options=["White", "Non-White"])

        st.subheader("Clinical Information")

        stroke_class = st.selectbox("Stroke Type",
                                    options=["ISCHEMIC", "HEMORRHAGIC", "TIA"],
                                    help="TIA = Transient Ischemic Attack")

        blood_pressure = st.selectbox(
            "Blood Pressure",
            options=["Normal", "Abnormal"],
            help="Normal: 91-129 systolic, 61-84 diastolic")

        marital_status = st.selectbox(
            "Marital Status",
            options=["MARRIED", "SINGLE", "DIVORCED", "WIDOWED"])

        readmissions = st.number_input("Number of Previous Readmissions",
                                       min_value=0,
                                       max_value=10,
                                       value=0)

        st.subheader("Clinical Severity (Optional)")
        st.caption("Leave at default if unknown")

        severity = st.slider("Severity Score", 0.0, 1.0, 0.1, 0.05)
        symptoms = st.slider("Sign/Symptom Score", 0.0, 1.0, 0.15, 0.05)
        comorbidities = st.slider("Comorbidity Score", 0.0, 1.0, 0.1, 0.05)

        submit_button = st.form_submit_button("🔍 Predict Risk",
                                              use_container_width=True)

    # Process form submission
    if submit_button:
        # Collect form data
        form_data = {
            'age_group': age_group,
            'sex': sex,
            'race': race,
            'stroke_class': stroke_class,
            'blood_pressure': blood_pressure,
            'marital_status': marital_status,
            'readmissions': readmissions,
            'pct_SEVERITY': severity,
            'pct_SIGN_SYMPTOM': symptoms,
            'pct_DISEASE_DISORDER': comorbidities
        }

        # Create feature vector
        patient_features = create_feature_dict(form_data, feature_names)

        # Predict risks
        with st.spinner("🔄 Computing risk predictions..."):
            risks = predict_risks(patient_features, models, feature_names)

        # Compute SHAP values
        with st.spinner("🔄 Analyzing risk factors..."):
            shap_values_dict = compute_shap_values(patient_features, models,
                                                   explainers, feature_names)
            top_features, top_values = get_top_risk_factors(shap_values_dict,
                                                            feature_names,
                                                            top_n=5)

        # Store in session state
        st.session_state.prediction_made = True
        st.session_state.risks = risks
        st.session_state.patient_features = patient_features
        st.session_state.form_data = form_data
        st.session_state.shap_values_dict = shap_values_dict
        st.session_state.top_features = top_features
        st.session_state.top_values = top_values

    # Display results if prediction was made
    if st.session_state.prediction_made:
        risks = st.session_state.risks
        patient_features = st.session_state.patient_features
        form_data = st.session_state.form_data
        shap_values_dict = st.session_state.shap_values_dict
        top_features = st.session_state.top_features
        top_values = st.session_state.top_values

        # Display results
        st.header("📊 Risk Assessment Results")

        # Risk summary boxes - dynamic number of columns
        sorted_horizons = sorted(risks.keys())
        num_horizons = len(sorted_horizons)
        cols = st.columns(num_horizons)

        for idx, h in enumerate(sorted_horizons):
            risk = risks[h]
            category, css_class = get_risk_category(risk)

            with cols[idx]:
                st.markdown(f"""
                <div class="risk-box {css_class}">
                    <h3>{h}-Day</h3>
                    <h2>{risk:.1%}</h2>
                    <p>{category} Risk</p>
                </div>
                """,
                            unsafe_allow_html=True)

        st.divider()

        # 1. Risk Trajectory Plot
        st.subheader("📈 1. Mortality Risk Trajectory")
        fig_trajectory = plot_risk_trajectory(risks)
        st.pyplot(fig_trajectory)

        st.divider()

        # 2. Top Risk Factors Plot
        st.subheader("🎯 2. Top Contributing Risk Factors")
        fig_factors = plot_top_risk_factors(top_features, top_values,
                                            patient_features)
        st.pyplot(fig_factors)

        st.divider()

        # 3. AI Explanation by Time Point
        st.subheader("🤖 3. AI-Powered Risk Explanation")

        selected_horizon = st.selectbox(
            "Select time horizon for detailed explanation:",
            options=sorted(risks.keys()),
            format_func=lambda x: f"{x}-Day Risk: {risks[x]:.1%}",
            key="horizon_selector")

        # Generate explanation button
        if st.button("Generate Explanation",
                     type="primary",
                     key="generate_explanation"):
            with st.spinner("🤖 Generating AI explanation..."):
                # Get SHAP values for selected horizon
                horizon_shap = shap_values_dict[selected_horizon]

                # Get top factors for this specific horizon
                top_indices = np.argsort(np.abs(horizon_shap))[-5:][::-1]
                horizon_top_features = [feature_names[i] for i in top_indices]
                horizon_top_shap = horizon_shap[top_indices]

                explanation = generate_llm_explanation(
                    form_data, risks[selected_horizon], selected_horizon,
                    horizon_top_features, horizon_top_shap, feature_names)

                st.info(explanation)

        # Download results
        st.divider()
        st.subheader("💾 Export Results")

        # Create summary report
        report = {
            'Patient_Age_Group': form_data['age_group'],
            'Sex': form_data['sex'],
            'Stroke_Type': form_data['stroke_class'],
            'Blood_Pressure': form_data['blood_pressure'],
            **{
                f'{h}_Day_Risk': f"{risks[h]:.3f}"
                for h in sorted(risks.keys())
            }
        }

        report_df = pd.DataFrame([report])

        col1, col2 = st.columns(2)
        with col1:
            csv = report_df.to_csv(index=False)
            st.download_button(label="📥 Download Risk Report (CSV)",
                               data=csv,
                               file_name="stroke_risk_assessment.csv",
                               mime="text/csv")

        # Add button to clear results and start over
        if st.button("🔄 New Patient Assessment", type="secondary"):
            st.session_state.prediction_made = False
            st.session_state.risks = None
            st.session_state.patient_features = None
            st.session_state.form_data = None
            st.session_state.shap_values_dict = None
            st.session_state.top_features = None
            st.session_state.top_values = None
            st.rerun()

    else:
        # Instructions when no prediction yet
        st.info(
            "👈 Please fill in the patient information form in the sidebar and click 'Predict Risk' to begin assessment."
        )

        st.subheader("ℹ️ How to Use This System")
        st.markdown("""
        1. **Enter Patient Data**: Fill in the patient demographics and clinical information in the sidebar
        2. **Predict Risk**: Click the 'Predict Risk' button to generate mortality risk predictions
        3. **Review Results**: 
           - View risk percentages at multiple time horizons
           - Examine the risk trajectory over time
           - Identify the top 5 contributing risk factors
        4. **Get AI Explanation**: Select a time horizon and generate a detailed clinical explanation
        5. **Export**: Download the risk assessment report for patient records
        
        **Note**: This is a decision support tool. All predictions should be reviewed by qualified healthcare professionals.
        """)

        st.subheader("📊 Model Performance")
        st.markdown(f"""
        - **Model Type**: LightGBM Gradient Boosting
        - **Training Date**: {model_bundle.get('train_date', 'N/A')}
        - **Version**: {model_bundle.get('version', 'N/A')}
        - **Time Horizons**: {', '.join([str(h) for h in horizons])} days
        """)


if __name__ == "__main__":
    main()
