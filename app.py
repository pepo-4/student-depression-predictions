"""
Streamlit Web App for Depression Risk Assessment.

Interactive quiz that:
1. Collects user input via form
2. Runs inference pipeline
3. Displays risk assessment with factors
"""

import streamlit as st
import pandas as pd
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
        background-color: #f5f5f5;
        border-radius: 0.3em;
        margin: 0.5em 0;
        font-size: 0.95em;
    }
    .factor-positive {
        border-left: 3px solid #d32f2f;
    }
    .factor-negative {
        border-left: 3px solid #388e3c;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE & PIPELINE
# ============================================================================

@st.cache_resource
def get_pipeline():
    """Load pipeline once and cache it."""
    return load_pipeline(models_dir='./models')

pipeline = get_pipeline()

# ============================================================================
# FEATURE OPTIONS (EXACT VALUES FROM TRAINING)
# ============================================================================

FEATURE_OPTIONS = {
    'Gender': ['Male', 'Female'],
    'Age': ['Basso_18-26', 'Alto_27-43'],
    'Academic Pressure': ['Basso', 'Medio', 'Alto'],
    'CGPA': ['Basso_5.03-6.65', 'Medio_6.69-8.4', 'Alto_8.42-10.0'],
    'Study Satisfaction': ['Basso', 'Medio', 'Alto'],
    'Sleep Duration': ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'],
    'Dietary Habits': ['Healthy', 'Moderate', 'Unhealthy'],
    'Work/Study Hours': ['Basso_0-4', 'Medio_5-9', 'Alto_10-12'],
    'Financial Stress': ['Basso', 'Medio', 'Alto'],
    'Family History of Mental Illness': ['Yes', 'No'],
    'Degree_Level': ['High School', 'Undergraduate', 'Postgraduate']
}

# ============================================================================
# MAIN UI
# ============================================================================

st.markdown('<div class="main-title">🧠 Depression Risk Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Answer the following questions about your lifestyle and mental health</div>', unsafe_allow_html=True)

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
                user_input[feature] = st.selectbox(
                    f"{feature}",
                    options=FEATURE_OPTIONS[feature],
                    key=f"select_{feature}"
                )
        else:
            with col2:
                user_input[feature] = st.selectbox(
                    f"{feature}",
                    options=FEATURE_OPTIONS[feature],
                    key=f"select_{feature}"
                )
    
    st.markdown("---")
    
    # Submit button
    submit_button = st.form_submit_button(
        label="📊 Calculate Risk Assessment",
        use_container_width=True
    )

# ============================================================================
# PREDICTION & RESULTS
# ============================================================================

if submit_button:
    try:
        # Run pipeline
        with st.spinner("🔄 Analyzing your profile..."):
            result = pipeline.predict_pipeline(user_input)
        
        # Display results
        st.success("✅ Analysis complete!")
        st.markdown("---")
        
        # Main risk display
        risk_class = "risk-high" if result['is_risk'] else "risk-low"
        st.markdown(
            f'<div class="{risk_class} risk-box">{result["message"]}</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # Detailed results in columns
        col1, col2, col3 = st.columns(3)
        
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
                value=result['cluster'],
                help=result['cluster_name']
            )
        
        with col3:
            st.metric(
                label="Log-Odds",
                value=f"{result['log_odds']:.2f}"
            )
        
        st.markdown("---")
        
        # Risk factors
        st.subheader("📋 Contributing Factors")
        
        factor_col1, factor_col2 = st.columns(2)
        
        # Positive factors (risk)
        with factor_col1:
            st.markdown("**🚩 Risk Factors (Positive)**")
            if result['positive_factors']:
                for factor_name, coef in result['positive_factors'][:3]:
                    clean_name = factor_name.replace('_', ' ')
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
            st.markdown("**🌿 Protective Factors (Negative)**")
            if result['negative_factors']:
                for factor_name, coef in result['negative_factors'][:3]:
                    clean_name = factor_name.replace('_', ' ')
                    st.markdown(
                        f'<div class="factor-box factor-negative">'
                        f'<strong>{clean_name}</strong><br/>'
                        f'Impact: {coef:.4f}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("No significant protective factors detected.")
        
        st.markdown("---")
        
        # Profile summary
        st.subheader("👤 Your Profile")
        profile_df = pd.DataFrame([user_input]).T
        profile_df.columns = ['Value']
        st.dataframe(profile_df, use_container_width=True)
        
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
