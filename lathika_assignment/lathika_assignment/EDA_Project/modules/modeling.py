"""
Modeling Module
Implements k-NN classification and k-Means clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')


def prepare_classification_data(df, target_col, feature_cols=None):
    """
    Prepare data for classification
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_col : str
        Target column for classification
    feature_cols : list
        Feature columns to use
    
    Returns:
    --------
    X, y : Features and target
    """
    if feature_cols is None:
        # Use numeric columns as features
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in feature_cols:
            feature_cols.remove(target_col)
    
    # Remove rows with missing values
    df_clean = df[feature_cols + [target_col]].dropna()
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Convert target to categorical if it's numeric
    if y.dtype in ['int64', 'float64']:
        # Create bins for classification
        y = pd.cut(y, bins=3, labels=['Low', 'Medium', 'High'])
    
    return X, y


def knn_classification(X, y, test_size=0.2, n_neighbors=5, random_state=42):
    """
    Implement k-Nearest Neighbors classification
    
    Parameters:
    -----------
    X : pd.DataFrame or np.array
        Features
    y : pd.Series or np.array
        Target variable
    test_size : float
        Proportion of test set
    n_neighbors : int
        Number of neighbors
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict
        Dictionary with model results
    """
    print("\n" + "="*80)
    print("K-NEAREST NEIGHBORS CLASSIFICATION")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(y.unique())}")
    
    # Train model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = knn.predict(X_test_scaled)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({(accuracy*100):.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("="*80 + "\n")
    
    return {
        'model': knn,
        'scaler': scaler,
        'accuracy': accuracy,
        'y_test': y_test,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }


def visualize_knn_results(results, save_path=None):
    """
    Visualize k-NN classification results
    
    Parameters:
    -----------
    results : dict
        Results from knn_classification
    save_path : str
        Path to save the plot
    """
    cm = results['confusion_matrix']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix - k-NN Classification', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"k-NN results visualization saved to {save_path}")
    
    plt.close()


def kmeans_clustering(X, n_clusters=3, random_state=42, max_iter=300):
    """
    Apply k-Means clustering to numeric data
    
    Parameters:
    -----------
    X : pd.DataFrame or np.array
        Features
    n_clusters : int
        Number of clusters
    random_state : int
        Random state for reproducibility
    max_iter : int
        Maximum iterations
    
    Returns:
    --------
    dict
        Dictionary with clustering results
    """
    print("\n" + "="*80)
    print("K-MEANS CLUSTERING")
    print("="*80)
    
    # Remove missing values
    X_clean = X.dropna() if isinstance(X, pd.DataFrame) else X[~np.isnan(X).any(axis=1)]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    print(f"\nData shape: {X_scaled.shape}")
    print(f"Number of clusters: {n_clusters}")
    
    # Apply k-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    
    print(f"\nClustering Results:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nCluster Sizes:")
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} points ({(count/len(labels)*100):.2f}%)")
    
    # Cluster centers
    print(f"\nCluster Centers (scaled):")
    centers = kmeans.cluster_centers_
    for i, center in enumerate(centers):
        print(f"  Cluster {i}: {center}")
    
    print("="*80 + "\n")
    
    return {
        'model': kmeans,
        'scaler': scaler,
        'labels': labels,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'X_scaled': X_scaled,
        'X_original': X_clean
    }


def visualize_clusters(results, feature_cols=None, save_path=None):
    """
    Visualize k-Means clusters using scatter plots
    
    Parameters:
    -----------
    results : dict
        Results from kmeans_clustering
    feature_cols : list
        Feature column names
    save_path : str
        Path to save the plot
    """
    X_scaled = results['X_scaled']
    labels = results['labels']
    
    # Use first two features for 2D visualization
    if X_scaled.shape[1] >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot with clusters
        scatter = axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, 
                                 cmap='viridis', alpha=0.6, s=50)
        axes[0].set_xlabel(f'Feature 1' if feature_cols is None else feature_cols[0], fontsize=12)
        axes[0].set_ylabel(f'Feature 2' if feature_cols is None else feature_cols[1], fontsize=12)
        axes[0].set_title('K-Means Clustering Results', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0])
        
        # Cluster centers
        centers = results['model'].cluster_centers_
        axes[0].scatter(centers[:, 0], centers[:, 1], c='red', marker='x', 
                       s=200, linewidths=3, label='Centroids')
        axes[0].legend()
        
        # Cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        axes[1].bar(unique, counts, color=plt.cm.viridis(np.linspace(0, 1, len(unique))))
        axes[1].set_xlabel('Cluster', fontsize=12)
        axes[1].set_ylabel('Number of Points', fontsize=12)
        axes[1].set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster visualization saved to {save_path}")
        
        plt.close()
    else:
        print("Need at least 2 features for visualization")


def find_optimal_clusters(X, max_clusters=10, random_state=42, save_path=None):
    """
    Find optimal number of clusters using elbow method and silhouette score
    
    Parameters:
    -----------
    X : pd.DataFrame or np.array
        Features
    max_clusters : int
        Maximum number of clusters to test
    random_state : int
        Random state for reproducibility
    save_path : str
        Path to save the plot (optional)
    
    Returns:
    --------
    dict
        Dictionary with optimal cluster information
    """
    X_clean = X.dropna() if isinstance(X, pd.DataFrame) else X[~np.isnan(X).any(axis=1)]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    
    # Find optimal k (elbow method approximation)
    # Calculate rate of change
    if len(inertias) > 1:
        diffs = np.diff(inertias)
        diff2 = np.diff(diffs)
        optimal_k_elbow = k_range[np.argmax(diff2) + 1] if len(diff2) > 0 else 3
    else:
        optimal_k_elbow = 3
    
    # Find optimal k (highest silhouette score)
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    
    print(f"\nOptimal number of clusters:")
    print(f"  Elbow method: {optimal_k_elbow}")
    print(f"  Silhouette method: {optimal_k_silhouette}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
    axes[0].axvline(x=optimal_k_elbow, color='r', linestyle='--', label=f'Optimal k={optimal_k_elbow}')
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inertia', fontsize=12)
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(k_range, silhouette_scores, marker='o', linewidth=2, markersize=8, color='green')
    axes[1].axvline(x=optimal_k_silhouette, color='r', linestyle='--', 
                   label=f'Optimal k={optimal_k_silhouette}')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score Method', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Optimal clusters visualization saved to {save_path}")
    
    plt.close()
    
    return {
        'optimal_k_elbow': optimal_k_elbow,
        'optimal_k_silhouette': optimal_k_silhouette,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }


if __name__ == "__main__":
    # Example usage
    data_path = "../data/real_estate_dataset.csv"
    df = pd.read_csv(data_path)
    
    # Prepare data for classification
    if 'price' in df.columns:
        # Create a target variable from price
        X, y = prepare_classification_data(df, 'price')
        
        # k-NN Classification
        knn_results = knn_classification(X, y, n_neighbors=5)
        visualize_knn_results(knn_results, save_path="../outputs/knn_confusion_matrix.png")
    
    # k-Means Clustering
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
    if len(numeric_cols) >= 2:
        X_cluster = df[numeric_cols]
        cluster_results = kmeans_clustering(X_cluster, n_clusters=3)
        visualize_clusters(cluster_results, feature_cols=numeric_cols[:2],
                          save_path="../outputs/kmeans_clusters.png")

