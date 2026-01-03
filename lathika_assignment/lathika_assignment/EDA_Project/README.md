# EDA Project - Real Estate Dataset Analysis

## Project Overview

This project performs a comprehensive Exploratory Data Analysis (EDA) on a real estate dataset, covering data import, cleaning, transformation, statistical analysis, visualization, and machine learning modeling.

## Project Structure

```
EDA_Project/
├── data/                          # CSV, Excel files
│   └── real_estate_dataset.csv
├── modules/                        # Python modules
│   ├── __init__.py
│   ├── data_import.py            # Data loading functionality
│   ├── data_cleaning.py          # Data cleaning and preprocessing
│   ├── transformation.py          # Data transformation and feature engineering
│   ├── stats_analysis.py         # Statistical analysis
│   ├── visualization.py          # Visualization functions
│   └── modeling.py               # Machine learning models
├── outputs/                       # Generated plots, reports, exported datasets
│   ├── *.png                     # Static visualizations
│   ├── *.html                    # Interactive visualizations
│   └── *.csv                     # Exported datasets
├── main.py                        # Main script to run the EDA project
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Features

### 1. Data Import
- Load dataset from CSV/Excel
- Display dataset structure and basic information

### 2. Data Export
- Export cleaned and processed datasets to CSV/Excel

### 3. Data Cleaning
- Handle missing values (imputation strategies)
- Remove duplicate records
- Identify and handle outliers using IQR method

### 4. Data Transformation
- Create derived features (price per sqft, bed/bath ratio, etc.)
- Normalize and standardize numeric features
- Encode categorical variables

### 5. Descriptive Statistics
- Calculate mean, median, mode, standard deviation
- Compute correlation and covariance matrices
- Frequency analysis for categorical variables
- Statistical interpretations

### 6. Basic Visualization
- Line plots
- Bar charts
- Histograms

### 7. Advanced Visualization
- Pair plots
- Correlation heatmaps
- Violin plots

### 8. Interactive Visualization
- Interactive scatter plots using Plotly
- Interactive dashboard with multiple visualizations

### 9. Probability Analysis
- Probability distribution visualizations
- Q-Q plots for normality testing

### 10. Modeling - Classification (k-NN)
- k-Nearest Neighbors classification
- Train-test split
- Model evaluation with accuracy and confusion matrix

### 11. Modeling - Clustering (k-Means)
- k-Means clustering
- Optimal cluster number detection (elbow method, silhouette score)
- Cluster visualization

### 12. Summary & Insights
- Comprehensive summary of findings
- Key insights and patterns

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to execute the complete EDA workflow:

```bash
python main.py
```

The script will:
1. Load the dataset
2. Perform all cleaning and transformation steps
3. Generate all visualizations
4. Run machine learning models
5. Save all outputs to the `outputs/` directory

## Output Files

All generated files are saved in the `outputs/` directory:

- **Visualizations**: PNG files for static plots, HTML files for interactive plots
- **Data Exports**: CSV files with cleaned, transformed, and clustered data
- **Statistics**: CSV files with descriptive statistics, correlation, and covariance matrices

## Dataset

The project uses a real estate dataset containing:
- Property URLs
- Number of bedrooms and bathrooms
- City and neighborhood information
- Property size and type
- Price information
- Listing dates

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn
- scipy

See `requirements.txt` for complete list with versions.

