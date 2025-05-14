"""
Heart Disease Prediction App using Streamlit
Based on the Framingham Heart Study dataset
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from translations import get_translation
from data_utils import (
    load_heart_disease_data, clean_data, preprocess_data,
    get_dataset_description, get_column_mapping, get_input_ranges
)
from model_utils import train_models, get_feature_importance, make_prediction
from visualization_utils import (
    display_correlation_heatmap, display_histograms, display_boxplots,
    display_pairplot, display_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_model_comparison, plot_feature_importance
)

# Function to translate text based on selected language
def t(key):
    """Translate text based on selected language"""
    return get_translation(key, st.session_state.language)

# Initialize session state for language
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state.data = None

# Initialize session state for models
if 'models_results' not in st.session_state:
    st.session_state.models_results = None

# Initialize session state for preprocessed data
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None

# Function to load and process data
def get_data():
    if st.session_state.data is None:
        with st.spinner(t('loading_data')):
            df = load_heart_disease_data()
            if df is not None:
                st.session_state.data = clean_data(df)
    return st.session_state.data

# Function to get preprocessed data for model training
def get_processed_data(df):
    if st.session_state.preprocessed_data is None and df is not None:
        with st.spinner(t('loading_data')):
            X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_data(df)
            st.session_state.preprocessed_data = {
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'scaler': scaler,
                'feature_names': feature_names
            }
    return st.session_state.preprocessed_data

# Function to train and get models
def get_models(X_train_scaled, X_test_scaled, y_train, y_test):
    if st.session_state.models_results is None and X_train_scaled is not None:
        with st.spinner("Training models..."):
            models_results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
            st.session_state.models_results = models_results
    return st.session_state.models_results

# App layout
def main():
    # Page config
    st.set_page_config(
        page_title=t('app_title'),
        page_icon="❤️",
        layout="wide"
    )
    
    # Language selector in sidebar
    st.sidebar.title(t('language'))
    selected_language = st.sidebar.radio(
        label="",
        options=["English", "العربية"],
        index=0 if st.session_state.language == 'en' else 1,
        horizontal=True
    )
    
    # Update language based on selection
    if selected_language == "English" and st.session_state.language != 'en':
        st.session_state.language = 'en'
        st.rerun()
    elif selected_language == "العربية" and st.session_state.language != 'ar':
        st.session_state.language = 'ar'
        st.rerun()
    
    # Navigation in sidebar
    st.sidebar.title(t('app_title'))
    page = st.sidebar.radio(
        label="",
        options=[
            t('nav_home'),
            t('nav_data_exploration'),
            t('nav_model_training'),
            t('nav_feature_importance'),
            t('nav_prediction')
        ]
    )
    
    # Disclaimer
    st.sidebar.markdown("---")
    st.sidebar.info(t('disclaimer'))
    
    # Main content based on selected page
    if page == t('nav_home'):
        render_home_page()
    elif page == t('nav_data_exploration'):
        render_data_exploration_page()
    elif page == t('nav_model_training'):
        render_model_training_page()
    elif page == t('nav_feature_importance'):
        render_feature_importance_page()
    elif page == t('nav_prediction'):
        render_prediction_page()

# Home page
def render_home_page():
    st.title(t('welcome_title'))
    st.markdown(t('welcome_description'))
    
    # About section
    st.header(t('about_section'))
    st.markdown(t('about_content'))
    
    # How to use section
    st.header(t('how_to_use'))
    st.markdown(t('how_to_use_content'))
    
    # Show dataset preview
    st.header(t('dataset_info'))
    df = get_data()
    if df is not None:
        st.write(df.head())

# Data exploration page
def render_data_exploration_page():
    st.title(t('data_exploration_title'))
    
    # Load data
    df = get_data()
    if df is None:
        st.error("Error loading dataset. Please check the file path.")
        return
    
    # Dataset information
    st.header(t('dataset_info'))
    
    # Dataset statistics
    st.subheader(t('dataset_stats'))
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write(f"**Missing values:** {df.isna().sum().sum()}")
    
    with col2:
        # Class distribution
        heart_disease_count = df['has_heart_disease'].value_counts()
        st.write(f"**No Heart Disease (0):** {heart_disease_count.get(0, 0)}")
        st.write(f"**Heart Disease (1):** {heart_disease_count.get(1, 0)}")
        st.write(f"**Heart Disease Percentage:** {heart_disease_count.get(1, 0) / df.shape[0] * 100:.2f}%")
    
    # Dataset description
    st.subheader(t('dataset_description'))
    description = get_dataset_description()
    st.markdown(description[st.session_state.language])
    
    # Data visualizations
    st.header(t('data_visualizations'))
    
    # Correlation matrix
    st.subheader(t('correlation_matrix'))
    fig_corr = display_correlation_heatmap(df)
    st.plotly_chart(fig_corr)
    
    # Feature distributions
    st.subheader(t('feature_distributions'))
    fig_hist = display_histograms(df)
    st.pyplot(fig_hist)
    
    # Feature boxplots
    st.subheader(t('feature_boxplots'))
    fig_box = display_boxplots(df)
    st.pyplot(fig_box)
    
    # Feature relationships (pairplot)
    st.subheader(t('feature_relationships'))
    fig_pair = display_pairplot(df)
    st.pyplot(fig_pair.fig)

# Model training page
def render_model_training_page():
    st.title(t('model_training_title'))
    st.markdown(t('training_description'))
    
    # Load data
    df = get_data()
    if df is None:
        st.error("Error loading dataset. Please check the file path.")
        return
    
    # Process data for modeling
    data = get_processed_data(df)
    if data is None:
        st.error("Error processing data for modeling.")
        return
    
    # Train models button
    train_button = st.button(t('train_models_button'))
    
    # Get or train models
    models_results = None
    if train_button or st.session_state.models_results is not None:
        models_results = get_models(
            data['X_train_scaled'],
            data['X_test_scaled'],
            data['y_train'],
            data['y_test']
        )
    
    # Display results if models are trained
    if models_results is not None:
        st.header(t('training_results'))
        
        # Model metrics comparison
        st.subheader(t('model_metrics'))
        fig_comparison = plot_model_comparison(models_results)
        st.plotly_chart(fig_comparison)
        
        # Confusion matrices
        st.subheader(t('confusion_matrices'))
        cols = st.columns(3)
        for i, (name, results) in enumerate(models_results.items()):
            with cols[i % 3]:
                fig_cm = display_confusion_matrix(
                    data['y_test'],
                    results['y_pred'],
                    name
                )
                st.plotly_chart(fig_cm)
        
        # ROC curves
        st.subheader(t('roc_curves'))
        fig_roc = plot_roc_curve(models_results)
        st.plotly_chart(fig_roc)
        
        # Precision-Recall curves
        st.subheader(t('pr_curves'))
        fig_pr = plot_precision_recall_curve(models_results)
        st.plotly_chart(fig_pr)

# Feature importance page
def render_feature_importance_page():
    st.title(t('feature_importance_title'))
    st.markdown(t('feature_importance_description'))
    
    # Load data
    df = get_data()
    if df is None:
        st.error("Error loading dataset. Please check the file path.")
        return
    
    # Process data for modeling
    data = get_processed_data(df)
    if data is None:
        st.error("Error processing data for modeling.")
        return
    
    # Get trained models
    models_results = get_models(
        data['X_train_scaled'],
        data['X_test_scaled'],
        data['y_train'],
        data['y_test']
    )
    
    if models_results is not None:
        # Extract feature importance
        feature_importance = get_feature_importance(
            models_results,
            data['feature_names']
        )
        
        # Display feature importance
        st.header(t('feature_importance_results'))
        
        if feature_importance:
            # Create tabs for each model's feature importance
            tabs = st.tabs(list(feature_importance.keys()))
            
            # Plot feature importance for each model
            importance_figs = plot_feature_importance(
                feature_importance,
                data['feature_names']
            )
            
            for i, (model_name, fig) in enumerate(importance_figs.items()):
                with tabs[i]:
                    st.plotly_chart(fig)
        else:
            st.warning("No feature importance available. Some models may not support feature importance extraction.")

# Prediction page
def render_prediction_page():
    st.title(t('prediction_title'))
    st.markdown(t('prediction_description'))
    
    # Load data
    df = get_data()
    if df is None:
        st.error("Error loading dataset. Please check the file path.")
        return
    
    # Process data for modeling
    data = get_processed_data(df)
    if data is None:
        st.error("Error processing data for modeling.")
        return
    
    # Get trained models
    models_results = get_models(
        data['X_train_scaled'],
        data['X_test_scaled'],
        data['y_train'],
        data['y_test']
    )
    
    if models_results is None:
        st.warning("Models need to be trained first. Please go to the Model Training page.")
        return
    
    # Get column mapping for display names
    column_mapping = get_column_mapping()[st.session_state.language]
    
    # Get input ranges for each feature
    input_ranges = get_input_ranges()
    
    # Create input form
    with st.form("prediction_form"):
        # Organize inputs into categories
        st.subheader(t('patient_information'))
        col1, col2, col3 = st.columns(3)
        with col1:
            male = st.radio(
                column_mapping['male'],
                options=[0, 1],
                format_func=lambda x: "Male" if x == 1 else "Female"
            )
        with col2:
            age = st.number_input(
                column_mapping['age'],
                min_value=input_ranges['age'][0],
                max_value=input_ranges['age'][1],
                value=50,
                step=input_ranges['age'][2]
            )
        with col3:
            education = st.selectbox(
                column_mapping['education'],
                options=[1, 2, 3, 4]
            )
        
        st.subheader(t('medical_history'))
        col1, col2, col3 = st.columns(3)
        with col1:
            current_smoker = st.radio(
                column_mapping['currentSmoker'],
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        with col2:
            cigs_per_day = st.number_input(
                column_mapping['cigsPerDay'],
                min_value=input_ranges['cigsPerDay'][0],
                max_value=input_ranges['cigsPerDay'][1],
                value=0 if current_smoker == 0 else 10,
                step=input_ranges['cigsPerDay'][2],
                disabled=(current_smoker == 0)
            )
        with col3:
            bp_meds = st.radio(
                column_mapping['BPMeds'],
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            prevalent_stroke = st.radio(
                column_mapping['prevalentStroke'],
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        with col2:
            prevalent_hyp = st.radio(
                column_mapping['prevalentHyp'],
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        with col3:
            diabetes = st.radio(
                column_mapping['diabetes'],
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        
        st.subheader(t('vital_signs'))
        col1, col2, col3 = st.columns(3)
        with col1:
            sys_bp = st.number_input(
                column_mapping['sysBP'],
                min_value=input_ranges['sysBP'][0],
                max_value=input_ranges['sysBP'][1],
                value=120,
                step=input_ranges['sysBP'][2]
            )
        with col2:
            dia_bp = st.number_input(
                column_mapping['diaBP'],
                min_value=input_ranges['diaBP'][0],
                max_value=input_ranges['diaBP'][1],
                value=80,
                step=input_ranges['diaBP'][2]
            )
        with col3:
            bmi = st.number_input(
                column_mapping['BMI'],
                min_value=input_ranges['BMI'][0],
                max_value=input_ranges['BMI'][1],
                value=25.0,
                step=input_ranges['BMI'][2],
                format="%.1f"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            heart_rate = st.number_input(
                column_mapping['heartRate'],
                min_value=input_ranges['heartRate'][0],
                max_value=input_ranges['heartRate'][1],
                value=75,
                step=input_ranges['heartRate'][2]
            )
        
        st.subheader(t('lab_results'))
        col1, col2 = st.columns(2)
        with col1:
            tot_chol = st.number_input(
                column_mapping['totChol'],
                min_value=input_ranges['totChol'][0],
                max_value=input_ranges['totChol'][1],
                value=200,
                step=input_ranges['totChol'][2]
            )
        with col2:
            glucose = st.number_input(
                column_mapping['glucose'],
                min_value=input_ranges['glucose'][0],
                max_value=input_ranges['glucose'][1],
                value=80,
                step=input_ranges['glucose'][2]
            )
        
        # Submit button
        submitted = st.form_submit_button(t('predict_button'))
    
    # Make prediction if form is submitted
    if submitted:
        # Prepare input data
        input_data = {
            'male': male,
            'age': age,
            'education': education,
            'currentSmoker': current_smoker,
            'cigsPerDay': cigs_per_day if current_smoker == 1 else 0,
            'BPMeds': bp_meds,
            'prevalentStroke': prevalent_stroke,
            'prevalentHyp': prevalent_hyp,
            'diabetes': diabetes,
            'totChol': tot_chol,
            'sysBP': sys_bp,
            'diaBP': dia_bp,
            'BMI': bmi,
            'heartRate': heart_rate,
            'glucose': glucose
        }
        
        # Make prediction
        predictions = make_prediction(
            input_data,
            models_results,
            data['scaler'],
            data['feature_names']
        )
        
        if predictions:
            # Display prediction results
            st.header(t('prediction_results'))
            
            # Create gauge chart for average prediction
            avg_prediction = predictions['Average']
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_prediction * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Heart Disease Risk (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 25], 'color': "green"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            st.plotly_chart(fig)
            
            # Display individual model predictions
            st.subheader("Model Predictions:")
            model_predictions = {k: v for k, v in predictions.items() if k != 'Average'}
            model_df = pd.DataFrame({
                'Model': list(model_predictions.keys()),
                'Risk (%)': [v * 100 for v in model_predictions.values()]
            })
            
            st.dataframe(model_df.style.format({'Risk (%)': '{:.2f}%'}))
            
            # Prediction explanation
            st.subheader(t('prediction_explanation'))
            if avg_prediction > 0.5:
                st.warning(t('high_risk_explanation'))
            else:
                st.success(t('low_risk_explanation'))
            
            # Risk factors
            st.subheader(t('risk_factors_identified'))
            
            # Identify top risk factors
            feature_importance = get_feature_importance(
                models_results,
                data['feature_names']
            )
            
            if feature_importance and 'Random Forest' in feature_importance:
                # Get top 5 features from Random Forest
                rf_importance = sorted(
                    feature_importance['Random Forest'],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                for feature, importance in rf_importance:
                    feature_display = column_mapping.get(feature, feature)
                    feature_value = input_data.get(feature)
                    
                    # Check if this is a concerning value
                    is_concerning = False
                    if feature == 'age' and feature_value > 55:
                        is_concerning = True
                    elif feature == 'sysBP' and feature_value > 140:
                        is_concerning = True
                    elif feature == 'diaBP' and feature_value > 90:
                        is_concerning = True
                    elif feature == 'BMI' and feature_value > 30:
                        is_concerning = True
                    elif feature == 'glucose' and feature_value > 100:
                        is_concerning = True
                    elif feature == 'totChol' and feature_value > 240:
                        is_concerning = True
                    elif feature in ['prevalentHyp', 'diabetes', 'currentSmoker'] and feature_value == 1:
                        is_concerning = True
                    
                    # Display with warning if concerning
                    if is_concerning:
                        st.warning(f"**{feature_display}:** {feature_value}")
                    else:
                        st.info(f"**{feature_display}:** {feature_value}")
            
            # Disclaimer
            st.info(t('disclaimer_prediction'))

if __name__ == "__main__":
    main()