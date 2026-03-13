"""
Data loading and preprocessing functions for heart disease dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_heart_disease_data(path: str = "data/heart_disease_uci.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("File not found. Please check the path.")
    except ValueError as e:
        print(f"If the CSV is empty or malformed: {e}")
    return None



# checking
# df = load_heart_disease_data()
# print(df)
# print(df.shape)

def preprocess_data(df):
   
    df = df.copy()

    # 1. Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # 2. Separate target if present
    target_col = "num"
    y = None
    if target_col in df.columns:
        y = df[target_col]
        df = df.drop(columns=[target_col])

    # 3. Identify numeric vs categorical columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(exclude=["int64", "float64"]).columns

    # 4. Handle missing values
    # numeric -> median, categorical -> mode[web:49][web:50][web:53]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # 5. Encode categorical variables (e.g., sex, cp, fbs, restecg, exang, slope, ca, thal, dataset)
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)  # one‑hot encoding

    # 6. Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # 7. Reattach target if it existed
    if y is not None:
        df[target_col] = y.values

    return df
# Testing
# df_raw = load_heart_disease_data()
# df_clean = preprocess_data(df_raw)

# print(df_clean.info())
# print(df_clean.head())


def prepare_regression_data(df, target='chol'):
    
    df = df.dropna(subset=[target])  # standard use of dropna with subset.

    # 2. Separate target
    y = df[target]

    # 3. Exclude target from features
    X = df.drop(columns=[target])  # typical pattern for creating X and y.

    return X, y

# df_raw = load_heart_disease_data()
# df_pre = preprocess_data(df_raw)   # from earlier step
# X_reg, y_reg = prepare_regression_data(df_pre, target="chol")


# Test
# print(X_reg.shape, y_reg.shape)
# print(X_reg.head())
# print(y_reg.head())


def prepare_classification_data(df, target='num'):

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    # 2. Binarize target:
    #    UCI heart disease: 0 = no disease, 1–4 = presence of disease[web:31][web:74][web:77]
    y = (df[target] > 0).astype(int)

    # 3. Define columns to drop from features
    cols_to_drop = [target]
    if "chol" in df.columns:      # exclude serum cholesterol from features
        cols_to_drop.append("chol")

    # 4. Build feature matrix X (all other columns)
    X = df.drop(columns=cols_to_drop)  # standard use of drop for feature selection[web:76][web:79]

    return X, y

# Test 
# df_raw = load_heart_disease_data()
# df_pre = preprocess_data(df_raw)

# X_cls, y_cls = prepare_classification_data(df_pre, target="num")
# print(X_cls.shape, y_cls.shape)
# print(y_cls.value_counts())


def split_and_scale(X, y, test_size=0.2, random_state=42):
    
     # 1. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y is not None else None,  # for classification balance
    )  # typical usage of train_test_split.

    # 2. Fit StandardScaler on training data only
    scaler = StandardScaler()  # standard preprocessing step.
    X_train_scaled = scaler.fit_transform(X_train)  # fit + transform on train.
    X_test_scaled = scaler.transform(X_test)        # transform only on test.

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
# Test
# df_raw = load_heart_disease_data()
# df_pre = preprocess_data(df_raw)
# X_cls, y_cls = prepare_classification_data(df_pre)

# X_train_s, X_test_s, y_train, y_test, scaler = split_and_scale(X_cls, y_cls)
# print(X_train_s.shape, X_test_s.shape)

