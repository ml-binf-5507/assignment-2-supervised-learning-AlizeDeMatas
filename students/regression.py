"""
Linear regression functions for predicting cholesterol using ElasticNet.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score


def train_elasticnet_grid(X_train, y_train, l1_ratios, alphas):

    results = []

    for l1 in l1_ratios:
        for alpha in alphas:
            model = ElasticNet(
                l1_ratio=l1,
                alpha=alpha,
                max_iter=5000,
                random_state=42,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            r2 = r2_score(y_train, y_pred)
            results.append(
                {
                    "l1_ratio": l1,
                    "alpha": alpha,
                    "r2_score": r2,
                    "model": model,
                }
            )

    return pd.DataFrame(results)
   
#quick sanity check
# np.random.seed(0)
# X = np.random.randn(20, 5)
# y = 2 * X[:, 0] - X[:, 1] + np.random.randn(20) * 0.1

# l1_ratios = [0.3, 0.7]
# alphas = [0.01, 0.1, 1.0]

# results = train_elasticnet_grid(X, y, l1_ratios, alphas)
# print(results.head())

def create_r2_heatmap(results_df, l1_ratios, alphas, output_path=None):
   
    pivot_df = results_df.pivot(
        index="l1_ratio",
        columns="alpha",
        values="r2_score",
    )

    # Ensure ordering matches the provided lists (optional but nice)
    pivot_df = pivot_df.reindex(index=l1_ratios)
    pivot_df = pivot_df.reindex(columns=alphas)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))

    # Heatmap with annotations
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "R² Score"},
        ax=ax,
    )  # seaborn.heatmap standard usage[web:25][web:47]

    # Axis labels and title
    ax.set_xlabel("Alpha")
    ax.set_ylabel("L1 Ratio")
    ax.set_title("ElasticNet R² scores")

    fig.tight_layout()

    # Save if output_path provided
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    return fig

# Generate the heatmap




def get_best_elasticnet_model(X_train, y_train, X_test, y_test, 
                               l1_ratios=None, alphas=None):
    
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

    # Train all models on training data
    results_df = train_elasticnet_grid(X_train, y_train, l1_ratios, alphas)

    best_model = None
    best_l1 = None
    best_alpha = None
    best_train_r2 = None
    best_test_r2 = -np.inf  # start below any possible R²[web:89]

    # Evaluate each trained model on the test set and pick best by test R²
    for _, row in results_df.iterrows():  # iterate over DataFrame rows[web:85][web:91]
        model = row["model"]
        y_test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)  # standard R² for regression[web:16][web:29][web:84]

        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_model = model
            best_l1 = row["l1_ratio"]
            best_alpha = row["alpha"]
            best_train_r2 = row["r2_score"]

    return {
        "model": best_model,
        "best_l1_ratio": best_l1,
        "best_alpha": best_alpha,
        "train_r2": best_train_r2,
        "test_r2": best_test_r2,
        "results_df": results_df,
    }

    
    
