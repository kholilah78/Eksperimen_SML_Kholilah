"""
automate_Kholilah-Nurafifah.py
================================
Script otomatisasi preprocessing dataset Heart Disease.
Melakukan seluruh tahapan preprocessing secara otomatis
dan menghasilkan dataset yang siap dilatih.

Author  : Kholilah Nurafifah
Dataset : Heart Disease UCI (Kaggle)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
RAW_DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'heart.csv')
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), 'heart_preprocessing')
TRAIN_OUT_PATH  = os.path.join(OUTPUT_DIR, 'heart_train.csv')
TEST_OUT_PATH   = os.path.join(OUTPUT_DIR, 'heart_test.csv')

TARGET_COL      = 'target'
CATEGORICAL_COLS = ['cp', 'restecg', 'slope', 'ca', 'thal']
SCALE_COLS      = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
OUTLIER_COLS    = ['trestbps', 'chol', 'thalach', 'oldpeak']
TEST_SIZE       = 0.2
RANDOM_STATE    = 42


# ─────────────────────────────────────────────
# FUNGSI-FUNGSI PREPROCESSING
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    print(f"[1/7] Loading data dari: {path}")
    df = pd.read_csv(path)
    print(f"      Shape awal: {df.shape}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris duplikat."""
    print("[2/7] Menghapus duplikat...")
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"      Dihapus: {n_before - len(df)} baris | Tersisa: {len(df)} baris")
    return df


def handle_missing_values(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Imputasi missing values dengan median (untuk kolom numerik)."""
    print("[3/7] Menangani missing values...")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [c for c in numerical_cols if c != target_col]

    imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    print(f"      Missing values tersisa: {df.isnull().sum().sum()}")
    return df


def handle_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Menangani outlier dengan metode IQR clipping."""
    print("[4/7] Menangani outlier (IQR clipping)...")
    df = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)
    print(f"      Kolom yang ditangani: {cols}")
    return df


def encode_categorical(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """One-Hot Encoding untuk fitur kategorikal."""
    print("[5/7] Encoding fitur kategorikal (One-Hot Encoding)...")
    existing_cols = [c for c in categorical_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cols, drop_first=True)
    print(f"      Shape setelah encoding: {df.shape}")
    return df


def split_and_scale(df: pd.DataFrame,
                    target_col: str,
                    scale_cols: list,
                    test_size: float = 0.2,
                    random_state: int = 42):
    """
    Train-test split dan feature scaling.

    Returns:
        X_train, X_test, y_train, y_test (DataFrames)
    """
    print("[6/7] Train-test split & feature scaling...")
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"      X_train: {X_train.shape} | X_test: {X_test.shape}")

    # Scale hanya kolom yang ada
    existing_scale = [c for c in scale_cols if c in X_train.columns]
    scaler = StandardScaler()
    X_train[existing_scale] = scaler.fit_transform(X_train[existing_scale])
    X_test[existing_scale]  = scaler.transform(X_test[existing_scale])
    print(f"      Kolom yang di-scale: {existing_scale}")

    return X_train, X_test, y_train, y_test


def save_results(X_train, X_test, y_train, y_test,
                 train_path: str, test_path: str) -> None:
    """Menyimpan hasil preprocessing ke file CSV."""
    print("[7/7] Menyimpan hasil preprocessing...")
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df  = pd.concat([X_test,  y_test],  axis=1)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    print(f"      Train disimpan ke : {train_path}")
    print(f"      Test  disimpan ke : {test_path}")


# ─────────────────────────────────────────────
# PIPELINE UTAMA
# ─────────────────────────────────────────────

def run_preprocessing(
    raw_path: str     = RAW_DATA_PATH,
    train_out: str    = TRAIN_OUT_PATH,
    test_out: str     = TEST_OUT_PATH,
    target_col: str   = TARGET_COL,
    cat_cols: list    = None,
    scale_cols: list  = None,
    outlier_cols: list = None,
    test_size: float  = TEST_SIZE,
    random_state: int = RANDOM_STATE
):
    """
    Jalankan seluruh pipeline preprocessing secara otomatis.

    Args:
        raw_path      : Path ke dataset mentah (.csv)
        train_out     : Path output dataset train
        test_out      : Path output dataset test
        target_col    : Nama kolom target
        cat_cols      : List kolom kategorikal
        scale_cols    : List kolom untuk scaling
        outlier_cols  : List kolom untuk penanganan outlier
        test_size     : Proporsi test set
        random_state  : Seed untuk reprodusibilitas

    Returns:
        dict berisi X_train, X_test, y_train, y_test
    """
    if cat_cols    is None: cat_cols    = CATEGORICAL_COLS
    if scale_cols  is None: scale_cols  = SCALE_COLS
    if outlier_cols is None: outlier_cols = OUTLIER_COLS

    print("=" * 55)
    print("  PIPELINE PREPROCESSING - HEART DISEASE DATASET")
    print("  Author: Kholilah Nurafifah")
    print("=" * 55)

    df = load_data(raw_path)
    df = remove_duplicates(df)
    df = handle_missing_values(df, target_col)
    df = handle_outliers_iqr(df, outlier_cols)
    df = encode_categorical(df, cat_cols)

    X_train, X_test, y_train, y_test = split_and_scale(
        df, target_col, scale_cols, test_size, random_state
    )
    save_results(X_train, X_test, y_train, y_test, train_out, test_out)

    print("=" * 55)
    print("  PREPROCESSING SELESAI!")
    print("=" * 55)

    return {
        'X_train': X_train,
        'X_test' : X_test,
        'y_train': y_train,
        'y_test' : y_test,
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    result = run_preprocessing()
    print(f"\nFitur yang dihasilkan ({result['X_train'].shape[1]} kolom):")
    print(list(result['X_train'].columns))
