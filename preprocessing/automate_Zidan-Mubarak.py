"""
Automated Wine Quality Data Preprocessing
Author: Zidan Mubarak
Description: Script untuk melakukan preprocessing otomatis pada Wine Quality Dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import sys


def load_data(file_path):
    """
    Load dataset dari file CSV
    
    Parameters:
    -----------
    file_path : str
        Path ke file CSV
        
    Returns:
    --------
    pd.DataFrame
        DataFrame yang berisi data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Data berhasil dimuat dari {file_path}")
        print(f"  Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"✗ Error: File {file_path} tidak ditemukan!")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error saat memuat data: {str(e)}")
        sys.exit(1)


def remove_duplicates(df):
    """
    Menghapus data duplikat
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame input
        
    Returns:
    --------
    pd.DataFrame
        DataFrame tanpa duplikat
    """
    initial_shape = df.shape[0]
    df_clean = df.drop_duplicates()
    removed = initial_shape - df_clean.shape[0]
    
    print(f"\n✓ Duplikat dihapus:")
    print(f"  Sebelum: {initial_shape} baris")
    print(f"  Sesudah: {df_clean.shape[0]} baris")
    print(f"  Dihapus: {removed} baris")
    
    return df_clean


def handle_outliers(df, method='cap'):
    """
    Menangani outliers menggunakan IQR method
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame input
    method : str
        Method untuk handle outliers ('cap' atau 'remove')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame setelah outliers ditangani
    """
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude Id and quality columns
    cols_to_process = [col for col in numeric_cols if col not in ['Id', 'quality']]
    
    print(f"\n✓ Menangani outliers dengan method: {method}")
    
    for col in cols_to_process:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if method == 'cap':
            # Capping outliers
            df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'remove':
            # Remove outliers
            df_processed = df_processed[
                (df_processed[col] >= lower_bound) & 
                (df_processed[col] <= upper_bound)
            ]
    
    print(f"  Shape setelah handle outliers: {df_processed.shape}")
    
    return df_processed


def feature_engineering(df):
    """
    Melakukan feature engineering
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame input
        
    Returns:
    --------
    pd.DataFrame
        DataFrame setelah feature engineering
    """
    df_processed = df.copy()
    
    print(f"\n✓ Feature Engineering:")
    
    # Drop Id column if exists
    if 'Id' in df_processed.columns:
        df_processed = df_processed.drop('Id', axis=1)
        print(f"  - Kolom 'Id' dihapus")
    
    print(f"  Jumlah fitur: {df_processed.shape[1]}")
    
    return df_processed


def scale_features(df, target_col='quality'):
    """
    Melakukan feature scaling menggunakan StandardScaler
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame input
    target_col : str
        Nama kolom target
        
    Returns:
    --------
    tuple
        (X_scaled_df, y, scaler)
    """
    print(f"\n✓ Feature Scaling:")
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Apply StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    print(f"  - Method: StandardScaler")
    print(f"  - Fitur yang di-scale: {len(X.columns)}")
    
    return X_scaled_df, y, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data menjadi training dan testing set
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    test_size : float
        Proporsi data testing
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    print(f"\n✓ Train-Test Split:")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  - Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f"  - Testing set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def save_processed_data(X, y, output_path):
    """
    Menyimpan data yang sudah diproses
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features yang sudah diproses
    y : pd.Series
        Target variable
    output_path : str
        Path untuk menyimpan file
    """
    # Combine X and y
    df_final = X.copy()
    df_final['quality'] = y.values
    
    # Save to CSV
    df_final.to_csv(output_path, index=False)
    
    print(f"\n✓ Data berhasil disimpan:")
    print(f"  Path: {output_path}")
    print(f"  Shape: {df_final.shape}")


def preprocess_pipeline(input_path, output_path, outlier_method='cap'):
    """
    Pipeline lengkap untuk preprocessing data
    
    Parameters:
    -----------
    input_path : str
        Path ke file input CSV
    output_path : str
        Path untuk menyimpan file output CSV
    outlier_method : str
        Method untuk handle outliers ('cap' atau 'remove')
        
    Returns:
    --------
    dict
        Dictionary berisi hasil preprocessing
    """
    print("=" * 60)
    print("AUTOMATED WINE QUALITY DATA PREPROCESSING")
    print("=" * 60)
    
    # 1. Load data
    df = load_data(input_path)
    
    # 2. Remove duplicates
    df_clean = remove_duplicates(df)
    
    # 3. Handle outliers
    df_no_outliers = handle_outliers(df_clean, method=outlier_method)
    
    # 4. Feature engineering
    df_processed = feature_engineering(df_no_outliers)
    
    # 5. Scale features
    X_scaled, y, scaler = scale_features(df_processed)
    
    # 6. Split data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # 7. Save processed data
    save_processed_data(X_scaled, y, output_path)
    
    print("\n" + "=" * 60)
    print("✓ PREPROCESSING SELESAI!")
    print("=" * 60)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'processed_data': (X_scaled, y)
    }


if __name__ == "__main__":
    # Default paths
    input_file = "../WineQT_raw/WineQT.csv"
    output_file = "WineQT_preprocessing.csv"
    
    # Check if custom paths are provided via command line
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Run preprocessing pipeline
    results = preprocess_pipeline(
        input_path=input_file,
        output_path=output_file,
        outlier_method='cap'
    )
    
    print(f"\n📊 Data siap untuk training!")
    print(f"   X_train shape: {results['X_train'].shape}")
    print(f"   X_test shape: {results['X_test'].shape}")
