"""
Diabetes Progression Prediction App
====================================
Interactive Streamlit application for predicting diabetes disease progression
"""

import streamlit as st
import diabetes_analysis as da
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px

#st.title("Diabetes Progression Prediction App")
# result = da.run_analysis()
#st.write(result)
# Page configuration
st.set_page_config(
    page_title="Diabetes Progression Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_model_artifacts():
    """Load saved model, scaler, and metadata"""
    try:
        model = joblib.load('models/model.joblib')
        scaler = joblib.load('models/preprocessor.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
        
        with open('models/model_card.json', 'r') as f:
            model_card = json.load(f)
        
        return model, scaler, feature_names, model_card
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None, None

# Engineering features
def engineer_features(data):
    """Create engineered features matching training"""
    df = data.copy()
    
    # Interaction features
    df['bmi_bp'] = df['bmi'] * df['bp']
    df['age_bmi'] = df['age'] * df['bmi']
    df['s1_s2'] = df['s1'] * df['s2']
    df['bmi_squared'] = df['bmi'] ** 2
    df['bp_squared'] = df['bp'] ** 2
    
    # Polynomial features for top 3
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[['bmi', 's5', 'bp']].values)
    
    for i in range(poly_features.shape[1]):
        df[f'poly_{i}'] = poly_features[:, i]
    
    return df

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Diabetes Progression Predictor</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        Predict disease progression using machine learning based on patient medical features
    </div>
    """, unsafe_allow_html=True)
    
    # Load artifacts
    model, scaler, feature_names, model_card = load_model_artifacts()
    
    if model is None:
        st.error("⚠️ Failed to load model. Please ensure model files exist in the 'models' directory.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/hospital.png", width=150)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page:",
            ["🎯 Make Prediction", "📊 Model Info", "📈 Feature Importance", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown(f"""
        ### Model Details
        - **Type**: {model_card['model_type']}
        - **Version**: {model_card['version']}
        - **Test R²**: {model_card['metrics']['test_r2']:.4f}
        - **RMSE**: {model_card['metrics']['rmse']:.2f}
        """)
    
    # Page routing
    if page == "🎯 Make Prediction":
        prediction_page(model, scaler, feature_names, model_card)
    elif page == "📊 Model Info":
        model_info_page(model_card)
    elif page == "📈 Feature Importance":
        feature_importance_page(model_card)
    else:
        about_page()

def prediction_page(model, scaler, feature_names, model_card):
    """Interactive prediction interface"""
    
    st.header("Enter Patient Information")
    
    # Brief model summary (uses model_card to avoid unused parameter warning)
    if model_card:
        try:
            test_r2 = model_card['metrics']['test_r2']
            st.info(f"Model: {model_card.get('model_name','Unknown')} (v{model_card.get('version','?')}) — Test R²: {test_r2:.3f}")
        except Exception:
            st.info(f"Model: {model_card.get('model_name','Unknown')} (v{model_card.get('version','?')})")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["🎚️ Sliders (Normalized)", "📝 Manual Entry", "📁 Upload CSV"]
    )
    if input_method == "🎚️ Sliders (Normalized)":
        # Slider inputs (normalized values)
        st.subheader("Adjust normalized feature values:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", -3.0, 3.0, 0.0, 0.1, 
                          help="Normalized age value")
            sex = st.slider("Sex", -3.0, 3.0, 0.0, 0.1,
                          help="Normalized sex value")
            bmi = st.slider("BMI", -3.0, 3.0, 0.0, 0.1,
                          help="Body Mass Index (normalized)")
            bp = st.slider("Blood Pressure", -3.0, 3.0, 0.0, 0.1,
                         help="Average blood pressure (normalized)")
            s1 = st.slider("S1 (TC)", -3.0, 3.0, 0.0, 0.1,
                         help="Total serum cholesterol")
        
        with col2:
            s2 = st.slider("S2 (LDL)", -3.0, 3.0, 0.0, 0.1,
                         help="Low-density lipoproteins")
            s3 = st.slider("S3 (HDL)", -3.0, 3.0, 0.0, 0.1,
                         help="High-density lipoproteins")
            s4 = st.slider("S4 (TCH)", -3.0, 3.0, 0.0, 0.1,
                         help="Total cholesterol / HDL")
            s5 = st.slider("S5 (LTG)", -3.0, 3.0, 0.0, 0.1,
                         help="Possibly log of serum triglycerides")
            s6 = st.slider("S6 (GLU)", -3.0, 3.0, 0.0, 0.1,
                         help="Blood sugar level")
        
        input_data = pd.DataFrame([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]], 
                                 columns=['age', 'sex', 'bmi', 'bp', 's1', 's2', 
                                        's3', 's4', 's5', 's6'])
    
    elif input_method == "📝 Manual Entry":
        st.info("Enter raw medical values (will be normalized automatically)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age_raw = st.number_input("Age (years)", 20, 80, 50)
            sex_raw = st.selectbox("Sex", ["Male", "Female"])
            bmi_raw = st.number_input("BMI", 15.0, 50.0, 25.0, 0.1)
            bp_raw = st.number_input("Blood Pressure", 60.0, 150.0, 90.0, 1.0)
            s1_raw = st.number_input("Total Cholesterol", 100, 300, 200)
        
        with col2:
            s2_raw = st.number_input("LDL Cholesterol", 50, 200, 100)
            s3_raw = st.number_input("HDL Cholesterol", 20, 100, 50)
            s4_raw = st.number_input("TCH", 2.0, 10.0, 4.0, 0.1)
            s5_raw = st.number_input("Triglycerides (log)", 3.0, 6.0, 4.5, 0.1)
            s6_raw = st.number_input("Blood Sugar", 70, 200, 90)
        
        # Normalize (simple normalization for demo)
        sex_val = 1.0 if sex_raw == "Male" else -1.0
        input_data = pd.DataFrame([[
            (age_raw - 50) / 15, sex_val, (bmi_raw - 26) / 5, 
            (bp_raw - 94) / 13, (s1_raw - 189) / 34, (s2_raw - 115) / 30,
            (s3_raw - 49) / 13, (s4_raw - 4.1) / 1.3, (s5_raw - 4.65) / 0.52,
            (s6_raw - 92) / 12
        ]], columns=['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])
    
    else:  # CSV Upload
        st.info("Upload a CSV file with columns: age, sex, bmi, bp, s1, s2, s3, s4, s5, s6")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            input_data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(input_data.head())
        else:
            st.warning("Please upload a CSV file to make predictions")
            return
    
    # Make prediction
    if st.button("🔮 Predict Disease Progression", type="primary"):
        with st.spinner("Making prediction..."):
            try:
                # Engineer features
                input_engineered = engineer_features(input_data)
                
                # Ensure correct feature order
                input_engineered = input_engineered[feature_names]
                
                # Scale features
                input_scaled = scaler.transform(input_engineered)
                
                # Predict
                prediction = model.predict(input_scaled)[0]
                
                # Display result
                st.markdown(f"""
                <div class='prediction-box'>
                    Predicted Disease Progression<br>
                    <span style='font-size: 3rem;'>{prediction:.1f}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Interpretation
                st.subheader("Interpretation")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediction", f"{prediction:.1f}")
                
                with col2:
                    percentile = (prediction - 25) / (346 - 25) * 100
                    st.metric("Percentile", f"{percentile:.0f}%")
                
                with col3:
                    if prediction < 100:
                        severity = "Low"
                        color = "🟢"
                    elif prediction < 200:
                        severity = "Moderate"
                        color = "🟡"
                    else:
                        severity = "High"
                        color = "🔴"
                    st.metric("Severity", f"{color} {severity}")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Disease Progression Score"},
                    gauge={
                        'axis': {'range': [0, 350]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 100], 'color': "lightgreen"},
                            {'range': [100, 200], 'color': "yellow"},
                            {'range': [200, 350], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': prediction
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature contribution (if available)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Key Contributing Features")
                    importances = model.feature_importances_
                    top_idx = np.argsort(importances)[-5:][::-1]
                    
                    fig = go.Figure(go.Bar(
                        x=[feature_names[i] for i in top_idx],
                        y=[importances[i] for i in top_idx],
                        marker_color='indianred'
                    ))
                    fig.update_layout(
                        title="Top 5 Important Features",
                        xaxis_title="Feature",
                        yaxis_title="Importance"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.error("Please ensure all features are provided correctly.")

def model_info_page(model_card):
    """Display model information"""
    st.header("📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Details")
        st.json({
            "Model Name": model_card['model_name'],
            "Model Type": model_card['model_type'],
            "Version": model_card['version'],
            "Created": model_card['created_date']
        })
        
        st.subheader("Dataset Info")
        st.json(model_card['dataset'])
    
    with col2:
        st.subheader("Performance Metrics")
        metrics = model_card['metrics']
        
        st.metric("Train R² Score", f"{metrics['train_r2']:.4f}")
        st.metric("Test R² Score", f"{metrics['test_r2']:.4f}")
        st.metric("RMSE", f"{metrics['rmse']:.2f}")
        st.metric("MAE", f"{metrics['mae']:.2f}")
        
        # Metrics visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Train R²', 'Test R²'],
            y=[metrics['train_r2'], metrics['test_r2']],
            marker_color=['lightblue', 'lightcoral']
        ))
        fig.update_layout(title="Model R² Scores", yaxis_title="R² Score")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Features")
    st.write(f"**Total Features**: {model_card['features']['total']}")
    
    tab1, tab2 = st.tabs(["Original Features", "Engineered Features"])
    
    with tab1:
        st.write(model_card['features']['original'])
    
    with tab2:
        st.write(model_card['features']['engineered'])
    
    st.subheader("Hyperparameters")
    st.json(model_card['hyperparameters'])

def feature_importance_page(model_card):
    """Display feature importance"""
    st.header("📈 Feature Importance Analysis")
    
    if model_card['feature_importance']:
        importance_df = pd.DataFrame(model_card['feature_importance'])
        
        fig = px.bar(
            importance_df, 
            x='importance', 
            y='feature',
            orientation='h',
            title='Top 10 Most Important Features',
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Feature Importance Table")
        st.dataframe(importance_df, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")

def about_page():
    """About page"""
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### Diabetes Progression Prediction System
    
    This application uses machine learning to predict diabetes disease progression
    based on patient medical features.
    
    #### Features:
    - 🎯 **Real-time Predictions**: Get instant disease progression estimates
    - 📊 **Interactive Inputs**: Multiple input methods (sliders, manual, CSV)
    - 📈 **Visual Analytics**: Comprehensive charts and metrics
    - 🔍 **Model Transparency**: Complete model information and performance
    
    #### How It Works:
    1. Enter patient medical information
    2. Features are automatically engineered and normalized
    3. ML model predicts disease progression score
    4. Results displayed with interpretation and severity level
    
    #### Dataset:
    The model is trained on the Diabetes dataset from scikit-learn, containing
    442 diabetes patients with 10 baseline medical features.
    
    #### Technology Stack:
    - **Framework**: Streamlit
    - **ML Library**: scikit-learn
    - **Visualization**: Plotly
    - **Model Tracking**: MLflow
    
    #### Created By : Ahmed Abbas
    SAIR ML Course - Final Project
    
    #### Resources:
    - [GitHub Repository](https://github.com/ahmedabbas358)
    - [Model Card](models/model_card.json)
    - [Documentation](README.md)
    
    ---
    
    ⚠️ **Disclaimer**: This is a demonstration project for educational purposes.
    Not intended for real medical diagnosis. Always consult healthcare professionals.
    """)

if __name__ == "__main__":
    main()