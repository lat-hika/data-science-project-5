"""
EDA Project Modules
"""

from .data_import import load_data, display_data_info
from .data_cleaning import handle_missing_values, remove_duplicates, identify_outliers, handle_outliers
from .transformation import normalize_data, standardize_data, create_derived_features
from .stats_analysis import calculate_descriptive_stats, interpret_statistics, calculate_correlation
from .visualization import *
from .modeling import knn_classification, kmeans_clustering, visualize_clusters

__all__ = [
    'load_data',
    'display_data_info',
    'handle_missing_values',
    'remove_duplicates',
    'identify_outliers',
    'handle_outliers',
    'normalize_data',
    'standardize_data',
    'create_derived_features',
    'calculate_descriptive_stats',
    'interpret_statistics',
    'calculate_correlation',
    'knn_classification',
    'kmeans_clustering',
    'visualize_clusters'
]

