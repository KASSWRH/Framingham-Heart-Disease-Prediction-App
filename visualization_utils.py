"""
Utilities for data visualization
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import pandas as pd

def display_correlation_heatmap(df):
    """
    Display correlation heatmap for numeric columns
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap using plotly
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title="Correlation Heatmap"
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=800,
    )
    
    return fig

def display_histograms(df, target_column='has_heart_disease'):
    """
    Display histograms for each feature grouped by the target variable
    """
    # Select only numeric columns excluding the target
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    # Create a figure
    fig = plt.figure(figsize=(20, 15))
    
    # Plot histograms for each feature
    for i, col in enumerate(numeric_cols):
        ax = fig.add_subplot(4, 4, i + 1)
        
        # Plot histograms grouped by target
        for target_val in [0, 1]:
            sns.histplot(
                data=df[df[target_column] == target_val],
                x=col,
                bins=15,
                alpha=0.5,
                label=f'Heart Disease: {target_val}',
                ax=ax
            )
        
        ax.set_title(f'Distribution of {col}')
        ax.legend()
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def display_boxplots(df, target_column='has_heart_disease'):
    """
    Display boxplots for each feature grouped by the target variable
    """
    # Select only numeric columns excluding the target
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    # Convert target to string for better visualization
    df_plot = df.copy()
    df_plot[target_column] = df_plot[target_column].map({0: 'No Heart Disease', 1: 'Heart Disease'})
    
    # Create subplots
    fig = plt.figure(figsize=(20, 15))
    
    for i, col in enumerate(numeric_cols):
        ax = fig.add_subplot(4, 4, i + 1)
        
        # Create boxplot
        sns.boxplot(
            data=df_plot,
            x=target_column,
            y=col,
            ax=ax
        )
        
        ax.set_title(f'Boxplot of {col}')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def display_pairplot(df, target_column='has_heart_disease'):
    """
    Display pairplot of important features
    """
    # Select only the most important features for pairplot
    important_features = ['age', 'sysBP', 'BMI', 'glucose', 'totChol', target_column]
    df_important = df[important_features]
    
    # Map target to string for better visualization
    df_important[target_column] = df_important[target_column].map({0: 'No Heart Disease', 1: 'Heart Disease'})
    
    # Create pairplot
    fig = sns.pairplot(
        data=df_important,
        hue=target_column,
        diag_kind='kde'
    )
    
    # Set title
    fig.fig.suptitle('Pairplot of Important Features', y=1.02, fontsize=16)
    
    return fig

def display_confusion_matrix(y_true, y_pred, model_name):
    """
    Display confusion matrix for a model
    """
    # Calculate confusion matrix
    cm = np.array([[np.sum((y_true == 0) & (y_pred == 0)), np.sum((y_true == 0) & (y_pred == 1))],
                  [np.sum((y_true == 1) & (y_pred == 0)), np.sum((y_true == 1) & (y_pred == 1))]])
    
    # Create heatmap using plotly
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual"),
        x=[0, 1],
        y=[0, 1],
        text_auto=True,
        color_continuous_scale='Blues',
        title=f"Confusion Matrix - {model_name}"
    )
    
    # Add annotations
    annotations = [
        dict(
            showarrow=False,
            text="True Negative",
            x=0,
            y=0,
            font=dict(color="white" if cm[0, 0] > cm.max() / 2 else "black")
        ),
        dict(
            showarrow=False,
            text="False Positive",
            x=1,
            y=0,
            font=dict(color="white" if cm[0, 1] > cm.max() / 2 else "black")
        ),
        dict(
            showarrow=False,
            text="False Negative",
            x=0,
            y=1,
            font=dict(color="white" if cm[1, 0] > cm.max() / 2 else "black")
        ),
        dict(
            showarrow=False,
            text="True Positive",
            x=1,
            y=1,
            font=dict(color="white" if cm[1, 1] > cm.max() / 2 else "black")
        )
    ]
    
    fig.update_layout(annotations=annotations)
    
    return fig

def plot_roc_curve(models_results):
    """
    Plot ROC curves for all models
    """
    # Create figure using plotly
    fig = go.Figure()
    
    # Add diagonal reference line (random classifier)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray'),
            showlegend=True
        )
    )
    
    # Add ROC curve for each model
    for name, results in models_results.items():
        fig.add_trace(
            go.Scatter(
                x=results['fpr'],
                y=results['tpr'],
                mode='lines',
                name=f"{name} (AUC={results['auc']:.3f})",
                showlegend=True
            )
        )
    
    # Update layout
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
        width=800,
        height=600
    )
    
    return fig

def plot_precision_recall_curve(models_results):
    """
    Plot Precision-Recall curves for all models
    """
    # Create figure using plotly
    fig = go.Figure()
    
    # Add Precision-Recall curve for each model
    for name, results in models_results.items():
        fig.add_trace(
            go.Scatter(
                x=results['recall_curve'],
                y=results['precision_curve'],
                mode='lines',
                name=f"{name} (AP={results['avg_precision']:.3f})",
                showlegend=True
            )
        )
    
    # Update layout
    fig.update_layout(
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend=dict(x=0.01, y=0.01, bgcolor='rgba(255, 255, 255, 0.5)'),
        width=800,
        height=600
    )
    
    return fig

def plot_model_comparison(models_results):
    """
    Plot model comparison bar chart
    """
    # Extract metrics for each model
    models = list(models_results.keys())
    accuracy = [results['accuracy'] for results in models_results.values()]
    precision = [results['precision'] for results in models_results.values()]
    recall = [results['recall'] for results in models_results.values()]
    f1 = [results['f1'] for results in models_results.values()]
    auc = [results['auc'] for results in models_results.values()]
    
    # Create a dataframe for the metrics
    metrics_df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc
    })
    
    # Melt the dataframe for easy plotting
    metrics_melted = pd.melt(metrics_df, id_vars=['Model'], var_name='Metric', value_name='Value')
    
    # Create the grouped bar chart
    fig = px.bar(
        metrics_melted,
        x='Model',
        y='Value',
        color='Metric',
        barmode='group',
        title='Model Performance Comparison',
        height=600
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Score",
        legend_title="Metric",
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def plot_feature_importance(feature_importance, feature_names):
    """
    Plot feature importance
    """
    # Create a figure for each model
    figs = {}
    
    for model_name, importance in feature_importance.items():
        # Sort feature importance
        sorted_importance = sorted(importance, key=lambda x: x[1], reverse=True)
        
        # Extract feature names and importance values
        features = [item[0] for item in sorted_importance]
        values = [item[1] for item in sorted_importance]
        
        # Create horizontal bar chart
        fig = px.bar(
            x=values,
            y=features,
            orientation='h',
            title=f"Feature Importance - {model_name}",
            labels={'x': 'Importance', 'y': 'Feature'},
            height=500
        )
        
        # Store figure
        figs[model_name] = fig
    
    return figs