"""
Data Import Module
Handles loading datasets from CSV, Excel, or SQL database
"""

import pandas as pd
import os


def load_data(file_path, file_type='csv'):
    """
    Load dataset from CSV, Excel, or SQL database
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
    file_type : str
        Type of file ('csv', 'excel', 'sql')
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'excel':
            df = pd.read_excel(file_path)
        elif file_type == 'sql':
            # For SQL, you would need connection parameters
            # This is a placeholder
            raise NotImplementedError("SQL loading not implemented in this example")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        print(f"Data loaded successfully from {file_path}")
        print(f"Dataset shape: {df.shape}")
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


def display_data_info(df, num_rows=5):
    """
    Display basic information about the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to display
    num_rows : int
        Number of rows to display
    """
    print("\n" + "="*80)
    print("DATASET INFORMATION")
    print("="*80)
    
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print(f"\nFirst {num_rows} rows:")
    print(df.head(num_rows))
    
    print(f"\nColumn Names:")
    print(df.columns.tolist())
    
    print(f"\nData Types:")
    print(df.dtypes)
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")
    
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    data_path = "../data/real_estate_dataset.csv"
    df = load_data(data_path, file_type='csv')
    display_data_info(df)

