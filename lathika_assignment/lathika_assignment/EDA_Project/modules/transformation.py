"""
Data Transformation Module
Handles normalization and standardization of numeric features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize numeric features using Min-Max scaling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to normalize
    columns : list
        Columns to normalize (None for all numeric columns)
    method : str
        Normalization method ('minmax', 'standard', 'robust')
    
    Returns:
    --------
    pd.DataFrame
        Dataset with normalized features
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_transformed = df.copy()
    
    print("\n" + "="*80)
    print("DATA NORMALIZATION")
    print("="*80)
    print(f"Method: {method}")
    print(f"Columns to normalize: {columns}")
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    for col in columns:
        if df_transformed[col].dtype in ['int64', 'float64']:
            # Store original values for reference
            original_min = df_transformed[col].min()
            original_max = df_transformed[col].max()
            original_mean = df_transformed[col].mean()
            original_std = df_transformed[col].std()
            
            # Transform
            df_transformed[col] = scaler.fit_transform(df_transformed[[col]])
            
            print(f"\n{col}:")
            print(f"  Original - Min: {original_min:.2f}, Max: {original_max:.2f}, Mean: {original_mean:.2f}, Std: {original_std:.2f}")
            print(f"  Transformed - Min: {df_transformed[col].min():.2f}, Max: {df_transformed[col].max():.2f}, Mean: {df_transformed[col].mean():.2f}, Std: {df_transformed[col].std():.2f}")
    
    print("="*80 + "\n")
    
    return df_transformed, scaler


def standardize_data(df, columns=None):
    """
    Standardize numeric features (mean=0, std=1)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to standardize
    columns : list
        Columns to standardize (None for all numeric columns)
    
    Returns:
    --------
    pd.DataFrame
        Dataset with standardized features
    """
    return normalize_data(df, columns, method='standard')


def create_derived_features(df):
    """
    Create derived features from existing columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to transform
    
    Returns:
    --------
    pd.DataFrame
        Dataset with derived features
    """
    df_transformed = df.copy()
    
    print("\n" + "="*80)
    print("CREATING DERIVED FEATURES")
    print("="*80)
    
    # Extract numeric size if possible
    if 'size' in df_transformed.columns:
        # Try to extract numeric size from size column
        def extract_size(value):
            if pd.isna(value):
                return np.nan
            if isinstance(value, (int, float)):
                return value
            # Extract first number from string
            import re
            numbers = re.findall(r'\d+', str(value))
            if numbers:
                return float(numbers[0])
            return np.nan
        
        df_transformed['size_numeric'] = df_transformed['size'].apply(extract_size)
        print("Created 'size_numeric' feature from 'size' column")
    
    # Price per square foot (if size_numeric exists)
    if 'size_numeric' in df_transformed.columns and 'price' in df_transformed.columns:
        df_transformed['price_per_sqft'] = df_transformed['price'] / (df_transformed['size_numeric'] + 1e-6)  # Add small value to avoid division by zero
        print("Created 'price_per_sqft' feature")
    
    # Bedroom to bathroom ratio
    if 'beds' in df_transformed.columns and 'baths' in df_transformed.columns:
        df_transformed['bed_bath_ratio'] = df_transformed['beds'] / (df_transformed['baths'] + 1e-6)
        print("Created 'bed_bath_ratio' feature")
    
    # Property type categories
    if 'type' in df_transformed.columns:
        df_transformed['is_apartment'] = df_transformed['type'].str.contains('Apartment|Flat', case=False, na=False).astype(int)
        df_transformed['is_villa'] = df_transformed['type'].str.contains('Villa|House', case=False, na=False).astype(int)
        df_transformed['is_land'] = df_transformed['type'].str.contains('Land|Plot', case=False, na=False).astype(int)
        print("Created property type indicator features")
    
    print("="*80 + "\n")
    
    return df_transformed


def encode_categorical_features(df, columns=None, method='onehot'):
    """
    Encode categorical features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to encode
    columns : list
        Categorical columns to encode
    method : str
        Encoding method ('onehot' or 'label')
    
    Returns:
    --------
    pd.DataFrame
        Dataset with encoded features
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    df_encoded = df.copy()
    
    print("\n" + "="*80)
    print("ENCODING CATEGORICAL FEATURES")
    print("="*80)
    print(f"Method: {method}")
    print(f"Columns to encode: {columns}")
    
    if method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, prefix=columns)
        print(f"Applied one-hot encoding to {columns}")
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in columns:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            print(f"Applied label encoding to {col}")
    
    print("="*80 + "\n")
    
    return df_encoded


if __name__ == "__main__":
    # Example usage
    data_path = "../data/real_estate_dataset.csv"
    df = pd.read_csv(data_path)
    
    # Create derived features
    df_transformed = create_derived_features(df)
    
    # Normalize numeric features
    df_normalized, scaler = normalize_data(df_transformed, method='minmax')

