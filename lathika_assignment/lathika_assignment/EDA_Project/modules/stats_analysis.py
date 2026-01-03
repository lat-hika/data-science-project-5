"""
Statistical Analysis Module
Calculates descriptive statistics and provides interpretations
"""

import pandas as pd
import numpy as np
from scipy import stats


def calculate_descriptive_stats(df, columns=None):
    """
    Calculate comprehensive descriptive statistics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    columns : list
        Columns to analyze (None for all numeric columns)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with descriptive statistics
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS")
    print("="*80)
    
    stats_dict = {}
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            data = df[col].dropna()
            
            stats_dict[col] = {
                'Count': len(data),
                'Mean': data.mean(),
                'Median': data.median(),
                'Mode': data.mode()[0] if not data.mode().empty else np.nan,
                'Std Dev': data.std(),
                'Variance': data.var(),
                'Min': data.min(),
                'Max': data.max(),
                'Range': data.max() - data.min(),
                'Q1 (25%)': data.quantile(0.25),
                'Q3 (75%)': data.quantile(0.75),
                'IQR': data.quantile(0.75) - data.quantile(0.25),
                'Skewness': stats.skew(data),
                'Kurtosis': stats.kurtosis(data)
            }
    
    stats_df = pd.DataFrame(stats_dict).T
    
    print("\nSummary Statistics:")
    print(stats_df.round(2))
    
    print("\n" + "="*80 + "\n")
    
    return stats_df


def interpret_statistics(stats_df):
    """
    Provide brief interpretation of key statistics
    
    Parameters:
    -----------
    stats_df : pd.DataFrame
        DataFrame with statistics
    """
    print("\n" + "="*80)
    print("STATISTICAL INTERPRETATION")
    print("="*80)
    
    for col in stats_df.index:
        mean = stats_df.loc[col, 'Mean']
        median = stats_df.loc[col, 'Median']
        std = stats_df.loc[col, 'Std Dev']
        skew = stats_df.loc[col, 'Skewness']
        
        print(f"\n{col}:")
        print(f"  Central Tendency: Mean = {mean:.2f}, Median = {median:.2f}")
        
        if abs(mean - median) < 0.1 * mean:
            print(f"  → Distribution appears approximately symmetric (mean ≈ median)")
        elif mean > median:
            print(f"  → Distribution is right-skewed (mean > median)")
        else:
            print(f"  → Distribution is left-skewed (mean < median)")
        
        print(f"  Variability: Standard Deviation = {std:.2f}")
        if std < 0.1 * mean:
            print(f"  → Low variability (std < 10% of mean)")
        elif std > 0.5 * mean:
            print(f"  → High variability (std > 50% of mean)")
        else:
            print(f"  → Moderate variability")
        
        print(f"  Skewness: {skew:.2f}")
        if abs(skew) < 0.5:
            print(f"  → Approximately symmetric distribution")
        elif skew > 0:
            print(f"  → Right-skewed distribution (tail on the right)")
        else:
            print(f"  → Left-skewed distribution (tail on the left)")
    
    print("="*80 + "\n")


def calculate_correlation(df, columns=None):
    """
    Calculate correlation matrix
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    columns : list
        Columns to include in correlation (None for all numeric columns)
    
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    corr_matrix = df[columns].corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # Find strong correlations
    print("\nStrong Correlations (|r| > 0.7):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                print(f"  {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_value:.3f}")
    
    print("="*80 + "\n")
    
    return corr_matrix


def calculate_covariance(df, columns=None):
    """
    Calculate covariance matrix
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    columns : list
        Columns to include in covariance (None for all numeric columns)
    
    Returns:
    --------
    pd.DataFrame
        Covariance matrix
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("\n" + "="*80)
    print("COVARIANCE ANALYSIS")
    print("="*80)
    
    cov_matrix = df[columns].cov()
    
    print("\nCovariance Matrix:")
    print(cov_matrix.round(2))
    
    print("="*80 + "\n")
    
    return cov_matrix


def frequency_analysis(df, columns=None):
    """
    Perform frequency analysis for categorical columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    columns : list
        Categorical columns to analyze
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    print("\n" + "="*80)
    print("FREQUENCY ANALYSIS")
    print("="*80)
    
    for col in columns:
        print(f"\n{col}:")
        freq = df[col].value_counts()
        print(freq.head(10))
        print(f"  Total unique values: {df[col].nunique()}")
        print(f"  Most common: {freq.index[0]} ({freq.iloc[0]} occurrences, {(freq.iloc[0]/len(df)*100):.2f}%)")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    data_path = "../data/real_estate_dataset.csv"
    df = pd.read_csv(data_path)
    
    # Calculate descriptive statistics
    stats_df = calculate_descriptive_stats(df)
    
    # Interpret statistics
    interpret_statistics(stats_df)
    
    # Calculate correlations
    corr_matrix = calculate_correlation(df)
    
    # Frequency analysis
    frequency_analysis(df)

