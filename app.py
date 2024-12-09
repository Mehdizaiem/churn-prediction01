import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

# Constants
MODELS_DIR = "models"

@st.cache_resource
def load_models():
    """
    Load all trained models from the models directory
    Returns a dictionary of model names and their loaded instances
    """
    models = {}
    try:
        # Load standard models
        for model_file in Path(MODELS_DIR).glob("*_model.joblib"):
            model_name = model_file.stem.replace('_model', '')
            models[model_name] = joblib.load(model_file)
        
        # Load CatBoost model separately if it exists
        catboost_path = Path(MODELS_DIR) / "CatBoost_model.cbm"
        if catboost_path.exists():
            from catboost import CatBoostClassifier
            models['CatBoost'] = CatBoostClassifier()
            models['CatBoost'].load_model(catboost_path)
            
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

@st.cache_resource
def load_model_info():
    """Load model feature information"""
    try:
        return joblib.load(Path(MODELS_DIR) / "model_info.joblib")
    except Exception as e:
        st.error(f"Error loading model info: {str(e)}")
        return None

def create_feature_input(feature, feature_type):
    """Create appropriate input widget based on feature type"""
    if feature in ['International plan', 'Voice mail plan']:
        return st.selectbox(feature, ['No', 'Yes'])
    elif 'int' in feature_type:
        return st.number_input(feature, value=0, step=1)
    else:
        return st.number_input(feature, value=0.0, step=0.1)

def main():
    st.set_page_config(
        page_title="Telecom Customer Churn Prediction",
        page_icon="📱",
        layout="wide"
    )
    
    st.title('📱 Telecom Customer Churn Prediction')
    
    # Load models and info
    models = load_models()
    model_info = load_model_info()
    
    if not models or not model_info:
        st.error("Failed to load necessary model files. Please check if all required files are present in the 'models' directory.")
        return
    
    # Sidebar - Model selection
    with st.sidebar:
        st.header('🔧 Model Settings')
        selected_model = st.selectbox(
            'Choose a model',
            options=list(models.keys()),
            help='Select the machine learning model to use for prediction'
        )
    
    # Main panel - Input features
    st.header('📝 Enter Customer Information')
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    # Create input fields based on features
    input_data = {}
    features = model_info['features']
    feature_types = model_info['feature_types']
    
    for i, feature in enumerate(features):
        # Alternate between columns
        with col1 if i % 2 == 0 else col2:
            input_data[feature] = create_feature_input(
                feature, 
                feature_types[feature]
            )
    
    # Make prediction
    if st.button('🔮 Predict', use_container_width=True):
        with st.spinner('Analyzing customer data...'):
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Get prediction
            model = models[selected_model]
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
            
            # Show results
            st.header('🎯 Prediction Results')
            
            # Create columns for the results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction[0]:
                    st.error('⚠️ Customer is likely to churn!')
                    st.write(f'Churn probability: {probability[0][1]:.1%}')
                else:
                    st.success('✅ Customer is likely to stay!')
                    st.write(f'Retention probability: {probability[0][0]:.1%}')
            
            with result_col2:
                # Create a gauge chart using HTML
                if prediction[0]:
                    churn_prob = probability[0][1] * 100
                    color = "red"
                else:
                    churn_prob = probability[0][0] * 100
                    color = "green"
                
                st.write(f"Confidence: {churn_prob:.1f}%")
                st.progress(churn_prob/100)

if __name__ == '__main__':
    main()