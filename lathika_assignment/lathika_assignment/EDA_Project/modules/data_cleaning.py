"""
Data Cleaning Module
Handles missing values, duplicates, and outliers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to clean
    strategy : str
        Strategy for imputation ('mean', 'median', 'mode', 'drop', 'forward_fill')
    columns : list
        Specific columns to handle (None for all columns)
    
    Returns:
    --------
    pd.DataFrame
        Dataset with missing values handled
    """
    df_cleaned = df.copy()
    
    if columns is None:
        columns = df_cleaned.columns
    
    print("\n" + "="*80)
    print("HANDLING MISSING VALUES")
    print("="*80)
    
    missing_before = df_cleaned[columns].isnull().sum()
    print(f"\nMissing values before cleaning:")
    print(missing_before[missing_before > 0] if missing_before.sum() > 0 else "No missing values")
    
    for col in columns:
        if df_cleaned[col].isnull().sum() > 0:
            if strategy == 'mean' and df_cleaned[col].dtype in ['int64', 'float64']:
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                print(f"Filled {col} with mean: {df_cleaned[col].mean():.2f}")
            
            elif strategy == 'median' and df_cleaned[col].dtype in ['int64', 'float64']:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                print(f"Filled {col} with median: {df_cleaned[col].median():.2f}")
            
            elif strategy == 'mode':
                mode_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else None
                if mode_value is not None:
                    df_cleaned[col].fillna(mode_value, inplace=True)
                    print(f"Filled {col} with mode: {mode_value}")
            
            elif strategy == 'drop':
                df_cleaned.dropna(subset=[col], inplace=True)
                print(f"Dropped rows with missing values in {col}")
            
            elif strategy == 'forward_fill':
                df_cleaned[col].ffill(inplace=True)
                print(f"Forward filled {col}")
    
    missing_after = df_cleaned[columns].isnull().sum()
    print(f"\nMissing values after cleaning:")
    print(missing_after[missing_after > 0] if missing_after.sum() > 0 else "No missing values")
    print("="*80 + "\n")
    
    return df_cleaned


def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to clean
    
    Returns:
    --------
    pd.DataFrame
        Dataset with duplicates removed
    """
    print("\n" + "="*80)
    print("REMOVING DUPLICATES")
    print("="*80)
    
    duplicates_before = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates_before}")
    
    df_cleaned = df.drop_duplicates()
    
    duplicates_after = df_cleaned.duplicated().sum()
    print(f"Number of duplicate rows after removal: {duplicates_after}")
    print(f"Rows removed: {duplicates_before - duplicates_after}")
    print("="*80 + "\n")
    
    return df_cleaned


def identify_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Identify outliers using IQR or Z-score method
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    columns : list
        Columns to check for outliers (None for all numeric columns)
    method : str
        Method to use ('iqr' or 'zscore')
    threshold : float
        Threshold for outlier detection
    
    Returns:
    --------
    dict
        Dictionary with outlier information
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("\n" + "="*80)
    print("OUTLIER DETECTION")
    print("="*80)
    
    outliers_info = {}
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > threshold]
                lower_bound = df[col].mean() - threshold * df[col].std()
                upper_bound = df[col].mean() + threshold * df[col].std()
            
            num_outliers = len(outliers)
            outliers_info[col] = {
                'count': num_outliers,
                'percentage': (num_outliers / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            print(f"\n{col}:")
            print(f"  Outliers: {num_outliers} ({(num_outliers/len(df)*100):.2f}%)")
            print(f"  Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    print("="*80 + "\n")
    
    return outliers_info


def handle_outliers(df, columns=None, method='cap', threshold=1.5):
    """
    Handle outliers by capping or removing them
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to clean
    columns : list
        Columns to handle outliers for
    method : str
        Method to use ('cap' or 'remove')
    threshold : float
        Threshold for outlier detection
    
    Returns:
    --------
    pd.DataFrame
        Dataset with outliers handled
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_cleaned = df.copy()
    
    print("\n" + "="*80)
    print("HANDLING OUTLIERS")
    print("="*80)
    
    for col in columns:
        if df_cleaned[col].dtype in ['int64', 'float64']:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            if method == 'cap':
                df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"Capped outliers in {col} to range [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            elif method == 'remove':
                before = len(df_cleaned)
                df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                after = len(df_cleaned)
                print(f"Removed {before - after} rows with outliers in {col}")
    
    print("="*80 + "\n")
    
    return df_cleaned


def visualize_outliers(df, columns=None, save_path=None):
    """
    Visualize outliers using box plots
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to visualize
    columns : list
        Columns to visualize
    save_path : str
        Path to save the plot
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        if idx < len(axes):
            axes[idx].boxplot(df[col].dropna())
            axes[idx].set_title(f'Box Plot: {col}')
            axes[idx].set_ylabel('Value')
    
    # Hide extra subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Outlier visualization saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    data_path = "../data/real_estate_dataset.csv"
    df = pd.read_csv(data_path)
    
    # Handle missing values
    df_cleaned = handle_missing_values(df, strategy='mean')
    
    # Remove duplicates
    df_cleaned = remove_duplicates(df_cleaned)
    
    # Identify outliers
    outliers = identify_outliers(df_cleaned, method='iqr')
    
    # Visualize outliers
    visualize_outliers(df_cleaned, save_path="../outputs/outliers_boxplot.png")

