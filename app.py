import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier

# Load models and info
@st.cache_resource
def load_models():
    models = {
        'GBM': joblib.load('models/GBM_model.joblib'),
        'Random Forest': joblib.load('models/RandomForest_model.joblib'),
        'XGBoost': joblib.load('models/XGBoost_model.joblib'),
        'Stacking': joblib.load('models/Stacking_model.joblib')
    }
    # Load CatBoost separately
    catboost = CatBoostClassifier()
    catboost.load_model('models/CatBoost_model.cbm')
    models['CatBoost'] = catboost
    
    return models

def main():
    st.title('Telecom Customer Churn Prediction')
    
    # Load models and info
    models = load_models()
    model_info = joblib.load('models/model_info.joblib')
    metrics = joblib.load('models/model_metrics.joblib')
    
    # Sidebar - Model selection
    st.sidebar.header('Select Model')
    selected_model = st.sidebar.selectbox('Choose a model', list(models.keys()))
    
    # Show model metrics
    st.sidebar.header('Model Performance')
    st.sidebar.write(f"Accuracy: {metrics[selected_model]['accuracy']:.3f}")
    st.sidebar.write(f"ROC-AUC: {metrics[selected_model]['roc_auc']:.3f}")
    
    # Main panel - Input features
    st.header('Enter Customer Information')
    
    # Create input fields for your features
    input_data = {}
    for feature in model_info['feature_names']:
        if feature in ['International plan', 'Voice mail plan']:
            input_data[feature] = st.selectbox(feature, ['Yes', 'No'])
        else:
            input_data[feature] = st.number_input(feature, value=0.0)
    
    # Make prediction
    if st.button('Predict'):
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Get prediction
        model = models[selected_model]
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        # Show results
        st.header('Prediction Results')
        if prediction[0]:
            st.error('Customer is likely to churn!')
            st.write(f'Churn probability: {probability[0][1]:.2%}')
        else:
            st.success('Customer is likely to stay!')
            st.write(f'Retention probability: {probability[0][0]:.2%}')

if __name__ == '__main__':
    main()