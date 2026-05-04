"""Streamlit deployment app for SVC Phishing model."""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Phishing Website Detector", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #c0392b; text-align: center; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 2rem;}
    .prediction-box {padding: 2rem; border-radius: 12px; text-align: center; margin: 1rem 0;}
    .stMetric {background: #f0f2f6; border-radius: 8px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🛡️ Phishing Website Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detect malicious URLs and websites using Support Vector Classification (SVC)</p>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    pipeline = joblib.load('model/svc_phishing_pipeline.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
    return pipeline, feature_names

try:
    pipeline, feature_names = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model not found. Please run the Jupyter notebook entirely to train and save the model to 'model/svc_phishing_pipeline.pkl'. Error: {e}")

if model_loaded:
    tab1, tab2 = st.tabs(["📝 Manual Input", "📁 Batch Predict (CSV)"])

    with tab1:
        st.markdown("### Enter Website Characteristics")
        st.info("Adjust the top indicators. Values are typically: **1 (Legitimate), 0 (Suspicious), -1 (Phishing)**. Non-specified features default to 1.")

        # Top features commonly associated with phishing in this dataset
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**URL & Domain Characteristics**")
            ssl_state = st.selectbox("SSL Final State", [1, 0, -1], index=0, help="1=Trusted, 0=Suspicious, -1=Untrusted/Missing")
            url_length = st.selectbox("URL Length", [1, 0, -1], index=0)
            prefix_suffix = st.selectbox("Prefix/Suffix in Domain (-)", [1, -1], index=0, help="Does the domain have a dash?")
            sub_domain = st.selectbox("Having Sub Domain", [1, 0, -1], index=0)

        with col2:
            st.markdown("**Page Content**")
            url_anchor = st.selectbox("URL of Anchor", [1, 0, -1], index=0, help="% of URLs pointing to different domains")
            links_tags = st.selectbox("Links in Tags", [1, 0, -1], index=0)
            sfh = st.selectbox("Server Form Handler (SFH)", [1, 0, -1], index=0)
            request_url = st.selectbox("Request URL", [1, -1], index=0)

        with col3:
            st.markdown("**Traffic & Metrics**")
            web_traffic = st.selectbox("Web Traffic", [1, 0, -1], index=0, help="Website ranking/traffic volume")
            age_domain = st.selectbox("Age of Domain", [1, -1], index=0, help="1 = >= 6 months")
            google_index = st.selectbox("Google Index", [1, -1], index=0)
            page_rank = st.selectbox("Page Rank", [1, -1], index=0)

        if st.button("🔮 Detect Phishing", type="primary", use_container_width=True):
            # Create a dictionary of all features initialized to 1 (Legitimate)
            input_dict = {feat: 1 for feat in feature_names}
            
            # Update with user inputs (mapping to exact names from the UCI dataset)
            # We map safely if the exact string matches
            mapping = {
                'SSLfinal_State': ssl_state,
                'URL_Length': url_length,
                'Prefix_Suffix': prefix_suffix,
                'having_Sub_Domain': sub_domain,
                'URL_of_Anchor': url_anchor,
                'Links_in_tags': links_tags,
                'SFH': sfh,
                'Request_URL': request_url,
                'web_traffic': web_traffic,
                'age_of_domain': age_domain,
                'Google_Index': google_index,
                'Page_Rank': page_rank
            }
            
            for k, v in mapping.items():
                if k in input_dict:
                    input_dict[k] = v
                    
            # Convert to DataFrame
            input_df = pd.DataFrame([input_dict])

            # Predict
            pred = pipeline.predict(input_df)[0]
            proba = pipeline.predict_proba(input_df)[0]
            
            if pred == 1:
                st.markdown(f"""
                <div class="prediction-box" style="background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); color: white;">
                    <h2>✅ Legitimate Website</h2>
                    <p style="font-size: 1.2rem;">Confidence: {proba[1]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box" style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); color: white;">
                    <h2>🚨 PHISHING DETECTED</h2>
                    <p style="font-size: 1.2rem;">Confidence: {proba[0]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Upload a CSV file")
        st.write("Upload a CSV file containing the 30 features expected by the model.")
        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                preds = pipeline.predict(df)
                
                result_df = df.copy()
                result_df['Prediction'] = ['Legitimate' if p == 1 else 'Phishing' for p in preds]
                
                st.success(f"Processed {len(preds)} websites!")
                st.dataframe(result_df[['Prediction'] + list(df.columns)], use_container_width=True)

                # Distribution
                st.markdown("### Detection Results")
                dist = result_df['Prediction'].value_counts()
                st.bar_chart(dist, color=["#2ecc71", "#e74c3c"])
                
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

    st.markdown("---")
    st.markdown("*Built with SVC (RBF kernel) + SelectKBest | Dataset: UCI Phishing Websites*")
