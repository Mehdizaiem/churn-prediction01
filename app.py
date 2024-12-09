import streamlit as st
import pandas as pd
import joblib
import os

@st.cache_resource
def load_models():
    """
    Load all trained models from the models directory
    Returns a dictionary of model names and their loaded instances
    """
    models = {}
    try:
        # List of expected model files
        model_files = [
            'GBM_model.joblib',
            'Random Forest_model.joblib',
            'XGBoost_model.joblib',
            'Stacking_model.joblib'
        ]
        
        # Load models
        for model_file in model_files:
            try:
                model_path = os.path.join('models', model_file)
                if os.path.exists(model_path):
                    model_name = model_file.replace('_model.joblib', '')
                    models[model_name] = joblib.load(model_path)
                else:
                    st.warning(f"Model file not found: {model_file}")
            except Exception as e:
                st.warning(f"Error loading {model_file}: {str(e)}")
        
        if not models:
            st.error("No models could be loaded. Please check model files.")
            return None
            
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Telecom Customer Churn Prediction",
        page_icon="📱",
        layout="wide"
    )
    
    st.title('📱 Telecom Customer Churn Prediction')
    
    # Load models and info
    models = load_models()
    model_info = joblib.load('models/model_info.joblib')
    
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
            if feature in ['International plan', 'Voice mail plan']:
                input_data[feature] = st.selectbox(feature, ['No', 'Yes'])
            else:
                input_data[feature] = st.number_input(feature, value=0.0)
    
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
                # Show confidence
                if prediction[0]:
                    churn_prob = probability[0][1] * 100
                else:
                    churn_prob = probability[0][0] * 100
                
                st.write(f"Confidence: {churn_prob:.1f}%")
                st.progress(churn_prob/100)

if __name__ == '__main__':
    main()