"""
automate_NamaSiswa.py
=====================
Script otomatisasi preprocessing dataset Titanic.
Mengembalikan data yang sudah siap dilatih (train & test split).

Cara pakai:
    python automate_NamaSiswa.py
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(source: str = "seaborn") -> pd.DataFrame:
    """
    Muat dataset Titanic.

    Parameters
    ----------
    source : str
        'seaborn'  → ambil langsung dari library seaborn
        'csv'      → baca dari file titanic_raw/titanic_raw.csv

    Returns
    -------
    pd.DataFrame  –  raw dataframe
    """
    if source == "seaborn":
        df = sns.load_dataset("titanic")
        print(f"[load_data] Dataset dimuat dari seaborn. Shape: {df.shape}")
    else:
        path = os.path.join("titanic_raw", "titanic_raw.csv")
        df = pd.read_csv(path)
        print(f"[load_data] Dataset dimuat dari {path}. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 2. SELEKSI FITUR
# ─────────────────────────────────────────────
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pilih kolom yang relevan untuk modeling.
    """
    features = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    df_sel = df[features].copy()
    print(f"[select_features] Fitur dipilih: {features}")
    return df_sel


# ─────────────────────────────────────────────
# 3. HANDLE MISSING VALUES
# ─────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tangani missing values:
    - age     → isi dengan median
    - embarked → isi dengan modus
    """
    before = df.isnull().sum().sum()

    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    after = df.isnull().sum().sum()
    print(f"[handle_missing] Missing values: {before} → {after}")
    return df


# ─────────────────────────────────────────────
# 4. ENCODING KATEGORIKAL
# ─────────────────────────────────────────────
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode kolom kategorikal:
    - sex      : male → 1, female → 0
    - embarked : C → 0, Q → 1, S → 2
    """
    le = LabelEncoder()
    df["sex"] = le.fit_transform(df["sex"])
    df["embarked"] = le.fit_transform(df["embarked"].astype(str))
    print("[encode_categorical] Encoding selesai: sex, embarked")
    return df


# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan fitur turunan:
    - family_size = sibsp + parch
    - is_alone    = 1 jika family_size == 0
    """
    df["family_size"] = df["sibsp"] + df["parch"]
    df["is_alone"] = (df["family_size"] == 0).astype(int)
    print("[feature_engineering] Fitur baru: family_size, is_alone")
    return df


# ─────────────────────────────────────────────
# 6. SCALING
# ─────────────────────────────────────────────
def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    StandardScaler pada fitur numerik kontinyu.
    """
    scaler = StandardScaler()
    cols = ["age", "fare", "family_size"]
    df[cols] = scaler.fit_transform(df[cols])
    print(f"[scale_features] Scaling diterapkan pada: {cols}")
    return df


# ─────────────────────────────────────────────
# 7. SPLIT & SIMPAN
# ─────────────────────────────────────────────
def split_and_save(
    df: pd.DataFrame,
    output_dir: str = "titanic_preprocessing",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Train-test split kemudian simpan ke CSV.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    X = df.drop("survived", axis=1)
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df["survived"] = y_train.values
    train_df.to_csv(os.path.join(output_dir, "titanic_train.csv"), index=False)

    test_df = X_test.copy()
    test_df["survived"] = y_test.values
    test_df.to_csv(os.path.join(output_dir, "titanic_test.csv"), index=False)

    print(f"[split_and_save] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[split_and_save] File disimpan di folder '{output_dir}'")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# PIPELINE UTAMA
# ─────────────────────────────────────────────
def run_preprocessing(
    source: str = "seaborn",
    output_dir: str = "titanic_preprocessing",
) -> tuple:
    """
    Jalankan seluruh pipeline preprocessing dari awal hingga akhir.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    print("=" * 50)
    print("  TITANIC PREPROCESSING PIPELINE")
    print("=" * 50)

    df = load_data(source)
    df = select_features(df)
    df = handle_missing(df)
    df = encode_categorical(df)
    df = feature_engineering(df)
    df = scale_features(df)
    result = split_and_save(df, output_dir)

    print("=" * 50)
    print("  PREPROCESSING SELESAI!")
    print("=" * 50)
    return result


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_preprocessing()
