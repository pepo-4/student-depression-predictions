"""
Streamlit Web App for Depression Risk Assessment.

Interactive quiz that:
1. Collects user input via form
2. Runs inference pipeline
3. Displays risk assessment with factors
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from utils import load_pipeline

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Depression Risk Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
        color: #1f77b4;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1em;
        color: #666;
        margin-bottom: 2em;
    }
    .risk-box {
        padding: 1.5em;
        border-radius: 0.5em;
        margin: 1em 0;
        font-size: 1.1em;
        font-weight: bold;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 4px solid #d32f2f;
        color: #b71c1c;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left: 4px solid #388e3c;
        color: #1b5e20;
    }
    .factor-box {
        padding: 0.8em;
        background: rgba(255, 255, 255, 0.92);
        border-radius: 0.3em;
        margin: 0.5em 0;
        font-size: 0.95em;
        color: #111111;
    }
    .factor-positive {
        border-left: 3px solid #d32f2f;
        background: rgba(211, 47, 47, 0.08);
    }
    .factor-negative {
        border-left: 3px solid #388e3c;
        background: rgba(56, 142, 60, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE & PIPELINE
# ============================================================================

@st.cache_resource
def get_pipeline(cache_version: str = "2026-05-06-3"):
    """Load pipeline once and cache it."""
    models_dir = Path(__file__).resolve().parent / 'models'
    return load_pipeline(models_dir=str(models_dir))

pipeline = get_pipeline()

# ============================================================================
# FEATURE OPTIONS (EXACT VALUES FROM TRAINING)
# ============================================================================

FEATURE_OPTIONS = {
    'Gender': ['Male', 'Female'],
    'Age': ['18-26 years', '27-43 years'],
    'Academic Pressure': ['High', 'Medium', 'Low'],
    'CGPA_30': ['15-20', '20-25', '25-30'],
    'Study Satisfaction': ['High', 'Medium', 'Low'],
    'Sleep Duration': ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'],
    'Dietary Habits': ['Healthy', 'Moderate', 'Unhealthy'],
    'Work/Study Hours': ['0-4 hours', '5-9 hours', '10-12 hours'],
    'Financial Stress': ['High', 'Medium', 'Low'],
    'Family History of Mental Illness': ['Yes', 'No'],
    'Degree_Level': ['High School', 'Undergraduate', 'Postgraduate']
}

# Mapping trentesimi (0-30) → CGPA (0-10)
CGPA_MAPPING = {
    '15-20': 'Basso_5.03-6.65',
    '20-25': 'Medio_6.69-8.4',
    '25-30': 'Alto_8.42-10.0'
}

PLACEHOLDER_OPTION = "Select..."


def with_placeholder(options):
    return [PLACEHOLDER_OPTION] + options


def encode_form_value(feature, value):
    if value == PLACEHOLDER_OPTION:
        return None
    if feature == 'Age':
        return 'Basso_18-26' if value == '18-26 years' else 'Alto_27-43'
    if feature in {'Academic Pressure', 'Study Satisfaction', 'Financial Stress'}:
        return {
            'High': 'Alto',
            'Medium': 'Medio',
            'Low': 'Basso'
        }[value]
    if feature == 'Work/Study Hours':
        return {
            '0-4 hours': 'Basso_0-4',
            '5-9 hours': 'Medio_5-9',
            '10-12 hours': 'Alto_10-12'
        }[value]
    return value


def format_factor_label(factor_name):
    feature_map = {
        'Age_': 'Age',
        'Academic_Pressure_': 'Academic Pressure',
        'Study_Satisfaction_': 'Study Satisfaction',
        'Financial_Stress_': 'Financial Stress',
        'Work_Study_Hours_': 'Work/Study Hours',
        'Sleep_Duration_': 'Sleep Duration',
        'Dietary_Habits_': 'Dietary Habits',
        'Family_History_of_Mental_Illness_': 'Family History of Mental Illness',
        'Degree_Level_': 'Degree Level',
        'Gender_': 'Gender',
        'CGPA_': 'CGPA'
    }

    value_map = {
        'Basso 18-26': '18-26 years',
        'Alto 27-43': '27-43 years',
        'Basso 0-4': '0-4 hours',
        'Medio 5-9': '5-9 hours',
        'Alto 10-12': '10-12 hours',
        'Alto 8.42-10.0': '25-30',
        'Medio 6.69-8.4': '20-25',
        'Basso 5.03-6.65': '15-20',
        'Basso': 'Low',
        'Medio': 'Medium',
        'Alto': 'High'
    }

    display_feature = factor_name
    display_value = factor_name

    for prefix, label in feature_map.items():
        if factor_name.startswith(prefix):
            display_feature = label
            display_value = factor_name[len(prefix):].replace('_', ' ')
            break

    display_value = value_map.get(display_value, display_value)

    return f"{display_feature} {display_value}".strip()

# ============================================================================
# MAIN UI
# ============================================================================

st.markdown('<div class="main-title">🧠 Depression Risk Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Answer the following questions about your lifestyle and mental health.</div>', unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# FORM
# ============================================================================

with st.form("assessment_form", clear_on_submit=False):
    
    # Create 2 columns for better layout
    col1, col2 = st.columns(2)
    
    user_input = {}
    
    # Distribute features across columns
    features_list = list(FEATURE_OPTIONS.keys())
    
    for i, feature in enumerate(features_list):
        if i % 2 == 0:
            with col1:
                if feature == 'CGPA_30':
                    cgpa_30_val = st.selectbox(
                        "CGPA",
                        options=with_placeholder(FEATURE_OPTIONS[feature]),
                        key=f"select_{feature}"
                    )
                    # Convert to 0-10 scale for model
                    user_input['CGPA'] = CGPA_MAPPING[cgpa_30_val] if cgpa_30_val != PLACEHOLDER_OPTION else None
                else:
                    label = f"{feature}"
                    if feature == 'Degree_Level':
                        label = "Degree Level (current)"
                    user_input[feature] = st.selectbox(
                        label,
                        options=with_placeholder(FEATURE_OPTIONS[feature]),
                        key=f"select_{feature}"
                    )
        else:
            with col2:
                if feature == 'CGPA_30':
                    cgpa_30_val = st.selectbox(
                        "CGPA",
                        options=with_placeholder(FEATURE_OPTIONS[feature]),
                        key=f"select_{feature}"
                    )
                    # Convert to 0-10 scale for model
                    user_input['CGPA'] = CGPA_MAPPING[cgpa_30_val] if cgpa_30_val != PLACEHOLDER_OPTION else None
                else:
                    label = f"{feature}"
                    if feature == 'Degree_Level':
                        label = "Degree Level (current)"
                    user_input[feature] = st.selectbox(
                        label,
                        options=with_placeholder(FEATURE_OPTIONS[feature]),
                        key=f"select_{feature}"
                    )
    
    st.markdown("---")
    
    # Submit button
    submit_button = st.form_submit_button(
        label="📊 Calculate Risk Assessment",
        use_container_width=True
    )

missing_fields = [field for field, value in user_input.items() if value is None or value == PLACEHOLDER_OPTION]

# ============================================================================
# PREDICTION & RESULTS
# ============================================================================

if submit_button and missing_fields:
    st.warning("Please complete all fields before starting the risk assessment.")
    st.stop()

if submit_button:
    try:
        model_input = {
            feature: encode_form_value(feature, value)
            for feature, value in user_input.items()
        }
        # Run pipeline
        with st.spinner("🔄 Analyzing your profile..."):
            result = pipeline.predict_pipeline(model_input)
        
        # Display results
        st.success("✅ Analysis complete!")
        st.markdown("---")
        
        # Detailed results in columns
        col0, col1, col2, col3 = st.columns(4)

        with col0:
            st.metric(
                label="Prediction",
                value=int(result['is_risk'])
            )
        
        with col1:
            st.metric(
                label="Probability",
                value=f"{result['probability']:.1%}",
                delta=f"Threshold: {result['threshold']:.1%}",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                label="Cluster Assignment",
                value=result['cluster_name']
            )
        
        with col3:
            train_prevalence = result.get('train_prevalence', result.get('threshold', 0.0))
            st.metric(
                label="Cluster depression rate",
                value=f"{train_prevalence:.1%}"
            )
        
        st.markdown("---")
        
        # Risk factors
        st.subheader("📋 Contributing Factors")
        
        factor_col1, factor_col2 = st.columns(2)
        
        # Positive factors (risk)
        with factor_col1:
            st.markdown("**🚩 Risk Factors**")
            if result['positive_factors']:
                for factor_name, coef in result['positive_factors'][:3]:
                    clean_name = format_factor_label(factor_name)
                    st.markdown(
                        f'<div class="factor-box factor-positive">'
                        f'<strong>{clean_name}</strong><br/>'
                        f'Impact: +{coef:.4f}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("No significant risk factors detected.")
        
        # Negative factors (protective)
        with factor_col2:
            st.markdown("**🌿 Protective Factors**")
            if result['negative_factors']:
                for factor_name, coef in result['negative_factors'][:3]:
                    clean_name = format_factor_label(factor_name)
                    st.markdown(
                        f'<div class="factor-box factor-negative">'
                        f'<strong>{clean_name}</strong><br/>'
                        f'Impact: {coef:.4f}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("No significant protective factors detected.")
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #999; font-size: 0.85em;">
    ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional mental health assessment.
    </div>
    """,
    unsafe_allow_html=True
)
