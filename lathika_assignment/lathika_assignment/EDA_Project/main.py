"""
Main Script for EDA Project
Orchestrates the complete EDA workflow
"""

import os
import sys
import pandas as pd
import numpy as np

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.data_import import load_data, display_data_info
from modules.data_cleaning import handle_missing_values, remove_duplicates, identify_outliers, handle_outliers, visualize_outliers
from modules.transformation import create_derived_features, normalize_data
from modules.stats_analysis import calculate_descriptive_stats, interpret_statistics, calculate_correlation, calculate_covariance, frequency_analysis
from modules.visualization import (
    create_histogram, create_bar_chart, create_line_plot,
    create_pair_plot, create_heatmap, create_violin_plot,
    create_probability_distribution, create_interactive_scatter, create_interactive_dashboard
)
from modules.modeling import (
    prepare_classification_data, knn_classification, visualize_knn_results,
    kmeans_clustering, visualize_clusters, find_optimal_clusters
)


def main():
    """
    Main function to run the complete EDA workflow
    """
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS PROJECT")
    print("Real Estate Dataset Analysis")
    print("="*80 + "\n")
    
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'outputs')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = os.path.join(data_dir, 'real_estate_dataset.csv')
    
    # ============================================================================
    # STEP 1: IMPORT DATA
    # ============================================================================
    print("\n" + "#"*80)
    print("STEP 1: IMPORT DATA")
    print("#"*80)
    
    df = load_data(data_path, file_type='csv')
    display_data_info(df, num_rows=10)
    
    # ============================================================================
    # STEP 2: EXPORT DATA (after any preprocessing)
    # ============================================================================
    print("\n" + "#"*80)
    print("STEP 2: EXPORT DATA")
    print("#"*80)
    
    # Save original data
    export_path = os.path.join(output_dir, 'original_data.csv')
    df.to_csv(export_path, index=False)
    print(f"Original data exported to: {export_path}")
    
    # ============================================================================
    # STEP 3: DATA CLEANING
    # ============================================================================
    print("\n" + "#"*80)
    print("STEP 3: DATA CLEANING")
    print("#"*80)
    
    # Handle missing values
    df_cleaned = handle_missing_values(df, strategy='mean')
    
    # Remove duplicates
    df_cleaned = remove_duplicates(df_cleaned)
    
    # Identify outliers
    outliers_info = identify_outliers(df_cleaned, method='iqr')
    
    # Visualize outliers
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        visualize_outliers(df_cleaned, columns=numeric_cols[:5], 
                         save_path=os.path.join(output_dir, 'outliers_boxplot.png'))
    
    # Handle outliers (cap method to preserve data)
    df_cleaned = handle_outliers(df_cleaned, method='cap', threshold=1.5)
    
    # ============================================================================
    # STEP 4: DATA TRANSFORMATION
    # ============================================================================
    print("\n" + "#"*80)
    print("STEP 4: DATA TRANSFORMATION")
    print("#"*80)
    
    # Create derived features
    df_transformed = create_derived_features(df_cleaned)
    
    # Export cleaned and transformed data
    export_path = os.path.join(output_dir, 'cleaned_data.csv')
    df_transformed.to_csv(export_path, index=False)
    print(f"\nCleaned and transformed data exported to: {export_path}")
    
    # ============================================================================
    # STEP 5: DESCRIPTIVE STATISTICS
    # ============================================================================
    print("\n" + "#"*80)
    print("STEP 5: DESCRIPTIVE STATISTICS")
    print("#"*80)
    
    stats_df = calculate_descriptive_stats(df_transformed)
    interpret_statistics(stats_df)
    
    # Save statistics
    stats_df.to_csv(os.path.join(output_dir, 'descriptive_statistics.csv'))
    
    # Correlation and covariance
    corr_matrix = calculate_correlation(df_transformed)
    corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
    
    cov_matrix = calculate_covariance(df_transformed)
    cov_matrix.to_csv(os.path.join(output_dir, 'covariance_matrix.csv'))
    
    # Frequency analysis
    frequency_analysis(df_transformed)
    
    # ============================================================================
    # STEP 6: BASIC VISUALIZATION
    # ============================================================================
    print("\n" + "#"*80)
    print("STEP 6: BASIC VISUALIZATION")
    print("#"*80)
    
    # Histogram
    if 'price' in df_transformed.columns:
        create_histogram(df_transformed, 'price', bins=30, 
                        title="Price Distribution", 
                        save_path=os.path.join(output_dir, 'price_histogram.png'))
    
    # Bar chart
    if 'city' in df_transformed.columns:
        create_bar_chart(df_transformed, 'city', title="Properties by City (Top 10)",
                        save_path=os.path.join(output_dir, 'city_bar_chart.png'))
    
    # Line plot (if we have a time series)
    if 'date' in df_transformed.columns and 'price' in df_transformed.columns:
        # Group by date
        try:
            df_transformed['date_parsed'] = pd.to_datetime(df_transformed['date'], errors='coerce')
            daily_price = df_transformed.groupby(df_transformed['date_parsed'].dt.date)['price'].mean().reset_index()
            if len(daily_price) > 1:
                create_line_plot(daily_price, 'date_parsed', 'price',
                               title="Average Price Over Time",
                               save_path=os.path.join(output_dir, 'price_trend_line.png'))
        except:
            pass
    
    # ============================================================================
    # STEP 7: ADVANCED VISUALIZATION
    # ============================================================================
    print("\n" + "#"*80)
    print("STEP 7: ADVANCED VISUALIZATION")
    print("#"*80)
    
    # Pair plot
    numeric_cols_for_pair = df_transformed.select_dtypes(include=[np.number]).columns.tolist()[:5]
    if len(numeric_cols_for_pair) >= 2:
        create_pair_plot(df_transformed, columns=numeric_cols_for_pair,
                        save_path=os.path.join(output_dir, 'pair_plot.png'))
    
    # Heatmap
    numeric_cols_for_heatmap = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols_for_heatmap) >= 2:
        create_heatmap(df_transformed, columns=numeric_cols_for_heatmap,
                      title="Correlation Heatmap",
                      save_path=os.path.join(output_dir, 'correlation_heatmap.png'))
    
    # Violin plot
    if 'city' in df_transformed.columns and 'price' in df_transformed.columns:
        # Select top cities for better visualization
        top_cities = df_transformed['city'].value_counts().head(5).index
        df_top_cities = df_transformed[df_transformed['city'].isin(top_cities)]
        create_violin_plot(df_top_cities, 'city', 'price',
                          title="Price Distribution by City",
                          save_path=os.path.join(output_dir, 'violin_plot.png'))
    
    # ============================================================================
    # STEP 8: INTERACTIVE VISUALIZATION
    # ============================================================================
    print("\n" + "#"*80)
    print("STEP 8: INTERACTIVE VISUALIZATION")
    print("#"*80)
    
    # Interactive scatter plot
    if 'size_numeric' in df_transformed.columns and 'price' in df_transformed.columns:
        create_interactive_scatter(
            df_transformed, 'size_numeric', 'price', 
            color_col='city' if 'city' in df_transformed.columns else None,
            title="Interactive Scatter: Price vs Size",
            save_path=os.path.join(output_dir, 'interactive_scatter.html')
        )
    
    # Interactive dashboard
    create_interactive_dashboard(
        df_transformed,
        save_path=os.path.join(output_dir, 'interactive_dashboard.html')
    )
    
    # ============================================================================
    # STEP 9: PROBABILITY ANALYSIS
    # ============================================================================
    print("\n" + "#"*80)
    print("STEP 9: PROBABILITY ANALYSIS")
    print("#"*80)
    
    # Probability distribution for price
    if 'price' in df_transformed.columns:
        create_probability_distribution(
            df_transformed, 'price',
            save_path=os.path.join(output_dir, 'price_probability_distribution.png')
        )
    
    # Probability distribution for size
    if 'size_numeric' in df_transformed.columns:
        create_probability_distribution(
            df_transformed, 'size_numeric',
            save_path=os.path.join(output_dir, 'size_probability_distribution.png')
        )
    
    # ============================================================================
    # STEP 10: MODELING - CLASSIFICATION (k-NN)
    # ============================================================================
    print("\n" + "#"*80)
    print("STEP 10: MODELING - CLASSIFICATION (k-NN)")
    print("#"*80)
    
    if 'price' in df_transformed.columns:
        # Prepare data for classification
        X, y = prepare_classification_data(df_transformed, 'price')
        
        if len(X) > 0 and len(y.unique()) > 1:
            # k-NN Classification
            knn_results = knn_classification(X, y, n_neighbors=5)
            visualize_knn_results(knn_results, 
                                save_path=os.path.join(output_dir, 'knn_confusion_matrix.png'))
        else:
            print("Insufficient data for classification")
    
    # ============================================================================
    # STEP 11: MODELING - CLUSTERING (k-Means)
    # ============================================================================
    print("\n" + "#"*80)
    print("STEP 11: MODELING - CLUSTERING (k-Means)")
    print("#"*80)
    
    # Select numeric features for clustering
    numeric_cols_cluster = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
    # Remove price if it exists (to avoid using target as feature)
    if 'price' in numeric_cols_cluster:
        numeric_cols_cluster.remove('price')
    
    if len(numeric_cols_cluster) >= 2:
        X_cluster = df_transformed[numeric_cols_cluster].dropna()
        
        if len(X_cluster) > 0:
            # Find optimal number of clusters
            optimal_clusters = find_optimal_clusters(X_cluster, max_clusters=8, 
                                                    save_path=os.path.join(output_dir, 'optimal_clusters.png'))
            
            # Use optimal number of clusters
            n_clusters = optimal_clusters['optimal_k_silhouette']
            
            # Apply k-Means
            cluster_results = kmeans_clustering(X_cluster, n_clusters=n_clusters)
            
            # Visualize clusters
            visualize_clusters(cluster_results, feature_cols=numeric_cols_cluster[:2],
                            save_path=os.path.join(output_dir, 'kmeans_clusters.png'))
            
            # Add cluster labels to dataframe
            df_transformed['cluster'] = np.nan
            df_transformed.loc[X_cluster.index, 'cluster'] = cluster_results['labels']
            
            # Save clustered data
            export_path = os.path.join(output_dir, 'clustered_data.csv')
            df_transformed.to_csv(export_path, index=False)
            print(f"\nClustered data exported to: {export_path}")
    
    # ============================================================================
    # STEP 12: SUMMARY & INSIGHTS
    # ============================================================================
    print("\n" + "#"*80)
    print("STEP 12: SUMMARY & INSIGHTS")
    print("#"*80)
    
    print("\n" + "="*80)
    print("KEY FINDINGS AND INSIGHTS")
    print("="*80)
    
    print(f"\n1. Dataset Overview:")
    print(f"   - Total records: {len(df)}")
    print(f"   - After cleaning: {len(df_transformed)}")
    print(f"   - Features: {len(df_transformed.columns)}")
    
    if 'price' in df_transformed.columns:
        print(f"\n2. Price Analysis:")
        print(f"   - Mean price: ₹{df_transformed['price'].mean():,.2f}")
        print(f"   - Median price: ₹{df_transformed['price'].median():,.2f}")
        print(f"   - Price range: ₹{df_transformed['price'].min():,.2f} - ₹{df_transformed['price'].max():,.2f}")
    
    if 'city' in df_transformed.columns:
        top_city = df_transformed['city'].value_counts().index[0]
        print(f"\n3. Geographic Distribution:")
        print(f"   - Most properties in: {top_city}")
        print(f"   - Number of cities: {df_transformed['city'].nunique()}")
    
    if 'type' in df_transformed.columns:
        top_type = df_transformed['type'].value_counts().index[0]
        print(f"\n4. Property Types:")
        print(f"   - Most common type: {top_type}")
        print(f"   - Number of types: {df_transformed['type'].nunique()}")
    
    if 'cluster' in df_transformed.columns:
        print(f"\n5. Clustering Results:")
        cluster_counts = df_transformed['cluster'].value_counts()
        for cluster, count in cluster_counts.items():
            print(f"   - Cluster {int(cluster)}: {count} properties ({(count/len(df_transformed)*100):.2f}%)")
    
    print("\n" + "="*80)
    print("\nAll outputs have been saved to the 'outputs' directory.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

