"""
Model evaluation functions: metrics and ROC/PR curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, auc as compute_auc, r2_score
)


def calculate_r2_score(y_true, y_pred):
    """
    Calculate R² score for regression.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True target values
    y_pred : np.ndarray or pd.Series
        Predicted target values
        
    Returns
    -------
    float
        R² score (between -inf and 1, higher is better)
    """
    return r2_score(y_true, y_pred)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred : np.ndarray or pd.Series
        Predicted binary labels
        
    Returns
    -------
    dict
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
   


def calculate_auroc_score(y_true, y_pred_proba):
    """
    Calculate Area Under the ROC Curve (AUROC).
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
        
    Returns
    -------
    float
        AUROC score (between 0 and 1)
    """
    return roc_auc_score(y_true, y_pred_proba)


def calculate_auprc_score(y_true, y_pred_proba):
    """
    Calculate Area Under the Precision-Recall Curve (AUPRC).
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
        
    Returns
    -------
    float
        AUPRC score (between 0 and 1)
    """
    return average_precision_score(y_true, y_pred_proba)


def generate_auroc_curve(y_true, y_pred_proba, model_name="Model", 
                        output_path=None, ax=None):
    """
    Generate and plot ROC curve.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
    model_name : str
        Name of the model (for legend)
    output_path : str, optional
        Path to save figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
        
    Returns
    -------
    tuple
        (figure, ax) or (figure,) if ax provided
    """
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Create axes if needed
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.figure

    # Plot ROC curve
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
    # Diagonal line
    ax.plot([0, 1], [0, 1], "k--", label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    if output_path is not None and created_fig:
        fig.tight_layout()
        fig.savefig(output_path)

    # Return figure and axes
    if created_fig:
        return fig, ax
    else:
        return fig, ax
    
    
def generate_auprc_curve(y_true, y_pred_proba, model_name="Model",
                        output_path=None, ax=None):
    """
    Generate and plot Precision-Recall curve.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
    model_name : str
        Name of the model (for legend)
    output_path : str, optional
        Path to save figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
        
    Returns
    -------
    tuple
        (figure, ax) or (figure,) if ax provided
    """
    # Precision-Recall curve and AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)

    # Create axes if needed
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.figure

    # Plot PR curve
    ax.plot(recall, precision, label=f"{model_name} (AP = {ap:.3f})")

    # Baseline: positive class prevalence
    prevalence = np.mean(y_true)
    ax.hlines(prevalence, 0, 1, colors="k", linestyles="--", label="Baseline")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")

    if output_path is not None and created_fig:
        fig.tight_layout()
        fig.savefig(output_path)

    return fig, ax


def plot_comparison_curves(y_true, y_pred_proba_log, y_pred_proba_knn,
                          output_path=None):
    """
    Plot ROC and PR curves for both logistic regression and k-NN side by side.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba_log : np.ndarray or pd.Series
        Predicted probabilities from logistic regression
    y_pred_proba_knn : np.ndarray or pd.Series
        Predicted probabilities from k-NN
    output_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with 2 subplots (ROC and PR curves)
    """
    # Create 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: ROC curves
    ax_roc = axes[0]
    generate_auroc_curve(
        y_true,
        y_pred_proba_log,
        model_name="Logistic Regression",
        ax=ax_roc,
    )
    generate_auroc_curve(
        y_true,
        y_pred_proba_knn,
        model_name="k-NN",
        ax=ax_roc,
    )
    ax_roc.set_title("ROC Curves")

    # Right: PR curves
    ax_pr = axes[1]
    generate_auprc_curve(
        y_true,
        y_pred_proba_log,
        model_name="Logistic Regression",
        ax=ax_pr,
    )
    generate_auprc_curve(
        y_true,
        y_pred_proba_knn,
        model_name="k-NN",
        ax=ax_pr,
    )
    ax_pr.set_title("Precision-Recall Curves")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)

    return fig
