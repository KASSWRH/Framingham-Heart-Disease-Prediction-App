"""
Utilities for training and evaluating machine learning models
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, average_precision_score

def train_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train multiple machine learning models and evaluate their performance
    """
    if X_train_scaled is None or y_train is None:
        return None
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
    }
    
    # Results dictionary
    models_results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        # Calculate Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        models_results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'avg_precision': avg_precision
        }
    
    return models_results

def get_feature_importance(models_results, feature_names):
    """
    Extract feature importance from models that support it
    """
    feature_importance = {}
    
    if models_results is None or feature_names is None:
        return feature_importance
    
    # Extract feature importance from models that support it
    if 'Random Forest' in models_results:
        rf_model = models_results['Random Forest']['model']
        feature_importance['Random Forest'] = list(zip(feature_names, rf_model.feature_importances_))
    
    if 'Gradient Boosting' in models_results:
        gb_model = models_results['Gradient Boosting']['model']
        feature_importance['Gradient Boosting'] = list(zip(feature_names, gb_model.feature_importances_))
    
    if 'AdaBoost' in models_results:
        ada_model = models_results['AdaBoost']['model']
        feature_importance['AdaBoost'] = list(zip(feature_names, ada_model.feature_importances_))
    
    if 'Logistic Regression' in models_results:
        lr_model = models_results['Logistic Regression']['model']
        # For logistic regression, use the absolute value of coefficients
        feature_importance['Logistic Regression'] = list(zip(feature_names, np.abs(lr_model.coef_[0])))
    
    return feature_importance

def make_prediction(input_data, models_results, scaler, feature_names):
    """
    Make predictions using all models for a single input
    """
    # Check if models are available
    if models_results is None or len(models_results) == 0:
        return None
    
    # Convert input data to DataFrame with proper column names
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Make predictions with each model
    predictions = {}
    for name, model_data in models_results.items():
        model = model_data['model']
        # Get probability of positive class
        prediction_proba = model.predict_proba(input_scaled)[0][1]
        predictions[name] = prediction_proba
    
    # Calculate average prediction
    avg_prediction = sum(predictions.values()) / len(predictions)
    predictions['Average'] = avg_prediction
    
    return predictions