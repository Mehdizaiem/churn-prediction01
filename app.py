import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Constants
MODELS_DIR = "models"

@st.cache_resource
def load_models():
    models = {}
    model_files = [
        'GBM_model.joblib',
        'Random Forest_model.joblib'
    ]
    
    for model_file in model_files:
        try:
            model_path = os.path.join('models', model_file)
            if os.path.exists(model_path):
                # Load the dictionary containing model and version info
                saved_dict = joblib.load(model_path)
                model_name = model_file.replace('_model.joblib', '')
                # Extract just the model
                models[model_name] = saved_dict['model']
                # Print version info for debugging
                st.write(f"Loading {model_name} trained with scikit-learn {saved_dict['created_with']['sklearn']}")
        except Exception as e:
            st.warning(f"Error loading {model_file}: {str(e)}")
            continue
    
    return models if models else None


def create_feature_input(feature, feature_type):
    """Create appropriate input widget based on feature type"""
    if feature in ['International plan', 'Voice mail plan']:
        return st.selectbox(feature, ['No', 'Yes'])
    else:
        # Force float32 type for numerical inputs
        val = st.number_input(
            feature,
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.2f"
        )
        return np.float32(val)

def process_input_data(input_data):
    """Process input data and convert to correct types"""
    # Convert Yes/No to 1/0 for categorical variables
    for key in ['International plan', 'Voice mail plan']:
        if key in input_data:
            input_data[key] = 1 if input_data[key] == 'Yes' else 0
    
    # Convert all numeric values to float32
    for key, value in input_data.items():
        if key not in ['International plan', 'Voice mail plan']:
            input_data[key] = np.float32(value)
    
    return input_data

def main():
    st.set_page_config(
        page_title="Telecom Customer Churn Prediction",
        page_icon="📱",
        layout="wide"
    )
    
    st.title('📱 Telecom Customer Churn Prediction')
    
    try:
        # Load models and info
        models = load_models()
        model_info = joblib.load(os.path.join('models', 'model_info.joblib'))
        
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
            try:
                with st.spinner('Analyzing customer data...'):
                    # Process input data
                    processed_data = process_input_data(input_data)
                    
                    # Convert to DataFrame with float32 dtype
                    input_df = pd.DataFrame([processed_data], dtype='float32')
                    
                    # Ensure column order matches training data
                    input_df = input_df[features]
                    
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
                        # Show confidence
                        if prediction[0]:
                            churn_prob = probability[0][1] * 100
                        else:
                            churn_prob = probability[0][0] * 100
                        
                        st.write(f"Confidence: {churn_prob:.1f}%")
                        st.progress(churn_prob/100)
                        
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        
if __name__ == '__main__':
    main()