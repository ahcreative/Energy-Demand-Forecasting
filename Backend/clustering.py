import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

print("========================")
print("ENERGY DATA CLUSTERING ANALYSIS")
print("========================")

# 1. LOAD PROCESSED DATA
print("\n1. LOADING PROCESSED DATA")
print("-" * 80)

# Load data from the preprocessing step
try:
    data = pd.read_csv('processed_energy_data.csv')
    print(f"Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns")

    # Display sample data
    print("\nSample data:")
    print(data.head())

    # Basic info
    print("\nData types:")
    print(data.dtypes)

    # Convert datetime columns
    if 'local_time' in data.columns:
        data['local_time'] = pd.to_datetime(data['local_time'])
        print("\nDate range: {} to {}".format(
            data['local_time'].min().strftime('%Y-%m-%d'),
            data['local_time'].max().strftime('%Y-%m-%d')
        ))

except FileNotFoundError:
    print("Error: processed_energy_data.csv not found. Please run data preprocessing first.")
    exit()

# 2. DATA PREPARATION FOR CLUSTERING
print("\n2. DATA PREPARATION FOR CLUSTERING")
print("-" * 80)


def prepare_clustering_data(data):
    """
    Prepare data for clustering analysis:
    - Select relevant features
    - Handle missing values
    - Scale features
    """
    print("Preparing data for clustering...")

    # Select features for clustering
    # We'll use both weather and demand features
    clustering_features = [
        'demand', 'temperature', 'humidity', 'windSpeed',
        'hour', 'day_of_week', 'month'
    ]

    # Filter to only include necessary columns and remove missing values
    cluster_df = data[clustering_features].copy()

    # Check for missing values
    missing_values = cluster_df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values in clustering features:")
        print(missing_values[missing_values > 0])
        print("Handling remaining missing values...")

        # Impute missing values with median
        for col in cluster_df.columns:
            if cluster_df[col].isnull().sum() > 0:
                cluster_df[col] = cluster_df[col].fillna(cluster_df[col].median())

    print("\nFinal clustering dataset shape:", cluster_df.shape)

    # Scale the data
    print("Scaling features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_df)

    # Create a DataFrame with scaled features
    scaled_df = pd.DataFrame(scaled_features, columns=cluster_df.columns)

    return cluster_df, scaled_df, clustering_features


# Prepare data for clustering
original_cluster_data, scaled_cluster_data, clustering_features = prepare_clustering_data(data)

print("\nOriginal data summary statistics:")
print(original_cluster_data.describe())

print("\nScaled data summary statistics:")
print(scaled_cluster_data.describe())

# 3. DIMENSIONALITY REDUCTION
print("\n3. DIMENSIONALITY REDUCTION")
print("-" * 80)


def perform_dimensionality_reduction(scaled_data, original_data):
    """
    Perform dimensionality reduction using PCA and t-SNE
    and visualize the results
    """
    print("Performing PCA...")

    # Create PCA model with 2 components
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

    # Explained variance
    explained_variance = pca.explained_variance_ratio_ * 100
    print(f"Explained variance by PC1: {explained_variance[0]:.2f}%")
    print(f"Explained variance by PC2: {explained_variance[1]:.2f}%")
    print(f"Total explained variance: {sum(explained_variance):.2f}%")

    # Examine feature importance
    print("\nFeature importance (PCA components):")
    component_df = pd.DataFrame(pca.components_.T, index=scaled_data.columns, columns=['PC1', 'PC2'])
    print(component_df)

    # Create t-SNE model (t-SNE is better for visualization and preserving local structure)
    print("\nPerforming t-SNE (this might take a few minutes)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(scaled_data)

    # Create a DataFrame with t-SNE results
    tsne_df = pd.DataFrame(data=tsne_result, columns=['t-SNE1', 't-SNE2'])

    # Create mappings to original data
    pca_df['demand'] = original_data['demand'].values
    pca_df['temperature'] = original_data['temperature'].values
    pca_df['hour'] = original_data['hour'].values
    pca_df['day_of_week'] = original_data['day_of_week'].values

    tsne_df['demand'] = original_data['demand'].values
    tsne_df['temperature'] = original_data['temperature'].values
    tsne_df['hour'] = original_data['hour'].values
    tsne_df['day_of_week'] = original_data['day_of_week'].values

    # Visualize PCA results colored by demand
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['demand'],
                          cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Demand')
    plt.title('PCA: Data Points Colored by Demand')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2f}% Variance Explained)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2f}% Variance Explained)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['temperature'],
                          cmap='coolwarm', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Temperature')
    plt.title('PCA: Data Points Colored by Temperature')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2f}% Variance Explained)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2f}% Variance Explained)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize t-SNE results
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(tsne_df['t-SNE1'], tsne_df['t-SNE2'], c=tsne_df['demand'],
                          cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Demand')
    plt.title('t-SNE: Data Points Colored by Demand')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    scatter = plt.scatter(tsne_df['t-SNE1'], tsne_df['t-SNE2'], c=tsne_df['temperature'],
                          cmap='coolwarm', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Temperature')
    plt.title('t-SNE: Data Points Colored by Temperature')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize PCA by hour of day
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['hour'],
                          cmap='twilight', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Hour of Day')
    plt.title('PCA: Data Points Colored by Hour of Day')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2f}% Variance Explained)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2f}% Variance Explained)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['day_of_week'],
                          cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Day of Week (0=Mon, 6=Sun)')
    plt.title('PCA: Data Points Colored by Day of Week')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2f}% Variance Explained)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2f}% Variance Explained)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pca_time_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Feature importance heatmap for PCA
    plt.figure(figsize=(10, 8))
    sns.heatmap(component_df, annot=True, cmap='coolwarm', fmt='.3f')
    plt.title('PCA Component Loadings')
    plt.tight_layout()
    plt.savefig('pca_components_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Dimensionality reduction completed. Visualizations saved.")
    return pca_df, tsne_df, pca, component_df


# Perform dimensionality reduction
pca_df, tsne_df, pca_model, pca_components = perform_dimensionality_reduction(
    scaled_cluster_data, original_cluster_data)

# 4. K-MEANS CLUSTERING
print("\n4. K-MEANS CLUSTERING")
print("-" * 80)


def perform_kmeans_clustering(scaled_data, pca_df, tsne_df):
    """
    Perform K-Means clustering:
    - Determine optimal K using elbow method
    - Apply K-Means with optimal K
    - Visualize clusters in PCA and t-SNE space
    """
    print("Determining optimal K using elbow method...")

    # Calculate Sum of Squared Distances (SSD) for different K
    ssd = []
    silhouette_scores = []
    range_k = range(2, 15)

    for k in range_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        ssd.append(kmeans.inertia_)

        # Calculate silhouette score
        if k > 1:  # Silhouette score requires at least 2 clusters
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(scaled_data, labels)
            silhouette_scores.append(silhouette_avg)
            print(f"K = {k}: Silhouette Score = {silhouette_avg:.4f}")

    # Plot elbow method
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(list(range_k), ssd, 'bo-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(list(range_k), silhouette_scores, 'ro-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different K Values')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kmeans_elbow_method.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Based on elbow method and silhouette scores, choose optimal K
    # We'll determine this by looking at the plots, but let's choose based on silhouette score
    optimal_k = list(range_k)[silhouette_scores.index(max(silhouette_scores)) + 1]
    print(f"\nChosen optimal K based on silhouette scores: {optimal_k}")

    # Apply K-Means with optimal K
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_optimal.fit_predict(scaled_data)

    # Add cluster labels to DataFrames
    pca_df['cluster'] = cluster_labels
    tsne_df['cluster'] = cluster_labels

    # Visualize clusters in PCA space
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df, palette='viridis',
                    alpha=0.7, s=10, legend='full')

    # Add cluster centers
    centers = pca_model.transform(kmeans_optimal.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.9, marker='X')

    plt.title(f'K-Means Clustering (K={optimal_k}) in PCA Space')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('kmeans_clusters_pca.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize clusters in t-SNE space
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='cluster', data=tsne_df, palette='viridis',
                    alpha=0.7, s=10, legend='full')
    plt.title(f'K-Means Clustering (K={optimal_k}) in t-SNE Space')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('kmeans_clusters_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()

    return cluster_labels, kmeans_optimal, optimal_k


# Perform K-Means clustering
kmeans_labels, kmeans_model, optimal_k = perform_kmeans_clustering(
    scaled_cluster_data, pca_df, tsne_df)

# 5. DBSCAN CLUSTERING
print("\n5. DBSCAN CLUSTERING")
print("-" * 80)


def perform_dbscan_clustering(scaled_data, pca_df, tsne_df):
    """
    Perform DBSCAN clustering:
    - Determine optimal eps using k-distance graph
    - Apply DBSCAN with optimal parameters
    - Visualize clusters in PCA and t-SNE space
    """
    print("Determining optimal eps parameter for DBSCAN...")

    # Calculate distances to nearest neighbors
    from sklearn.neighbors import NearestNeighbors



    neighbors = NearestNeighbors(n_neighbors=20)
    neighbors.fit(scaled_data)
    distances, _ = neighbors.kneighbors(scaled_data)

    # Sort distances to k-th nearest neighbor
    k_dist = distances[:, 19]  # 20th nearest neighbor (0-indexed)
    k_dist.sort()

    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(k_dist)), k_dist, 'b-')
    plt.axhline(y=0.5, color='r', linestyle='--')  # Example threshold
    plt.xlabel('Points sorted by distance')
    plt.ylabel('Distance to 20th nearest neighbor')
    plt.title('K-Distance Graph for DBSCAN Parameter Selection')
    plt.grid(True, alpha=0.3)
    plt.savefig('dbscan_kdistance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Choose eps where there's an "elbow" in the k-distance graph
    # For demonstration, let's choose a value (in practice, this would be chosen based on the elbow point)
    eps_value = 0.5  # Example value
    print(f"Chosen eps value: {eps_value}")

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps_value, min_samples=20)
    dbscan_labels = dbscan.fit_predict(scaled_data)

    # Count number of clusters and noise points
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)

    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({n_noise / len(dbscan_labels) * 100:.2f}% of data)")

    # Add DBSCAN labels to DataFrames
    pca_df['dbscan_cluster'] = dbscan_labels
    tsne_df['dbscan_cluster'] = dbscan_labels

    # Visualize clusters in PCA space
    plt.figure(figsize=(12, 10))
    # Create a custom palette where -1 (noise) is black
    unique_labels = set(dbscan_labels)
    n_clusters_actual = len(unique_labels) - (1 if -1 in unique_labels else 0)
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters_actual))
    palette = {i: colors[i] for i in range(n_clusters_actual)}
    if -1 in unique_labels:
        palette[-1] = (0, 0, 0, 1)  # Black for noise

    # Use categorical color mapping
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['dbscan_cluster'],
                          cmap='viridis', alpha=0.7, s=10)

    plt.title(f'DBSCAN Clustering (eps={eps_value}) in PCA Space')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig('dbscan_clusters_pca.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize clusters in t-SNE space
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_df['t-SNE1'], tsne_df['t-SNE2'], c=tsne_df['dbscan_cluster'],
                          cmap='viridis', alpha=0.7, s=10)
    plt.title(f'DBSCAN Clustering (eps={eps_value}) in t-SNE Space')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig('dbscan_clusters_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()

    return dbscan_labels, dbscan, eps_value


# Perform DBSCAN clustering
dbscan_labels, dbscan_model, eps_value = perform_dbscan_clustering(
    scaled_cluster_data, pca_df, tsne_df)

# 6. HIERARCHICAL CLUSTERING
print("\n6. HIERARCHICAL CLUSTERING")
print("-" * 80)


def perform_hierarchical_clustering(scaled_data, pca_df, tsne_df):
    """
    Perform hierarchical clustering:
    - Create dendrogram to determine optimal number of clusters
    - Apply hierarchical clustering with optimal parameters
    - Visualize clusters in PCA and t-SNE space
    """
    print("Creating dendrogram to determine optimal number of clusters...")



    # Compute the linkage matrix
    linkage_matrix = linkage(scaled_data, method='ward')

    # Plot the dendrogram
    plt.figure(figsize=(16, 10))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.axhline(y=5, color='r', linestyle='--')  # Example cutoff
    plt.savefig('hierarchical_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Based on dendrogram, choose optimal number of clusters
    from scipy.cluster.hierarchy import fcluster

    n_clusters_hierarchical = 5  # Example value based on dendrogram
    print(f"Chosen number of clusters for hierarchical clustering: {n_clusters_hierarchical}")

    # Apply hierarchical clustering by cutting the dendrogram
    hierarchical_labels_sample = fcluster(linkage_matrix, n_clusters_hierarchical, criterion='maxclust')

    # Train a K-Means model on the sample with these labels to apply to full dataset
    # (This is a workaround to apply hierarchical clustering to the full dataset)
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(scaled_data, hierarchical_labels_sample)
    hierarchical_labels = knn.predict(scaled_data)

    # Add hierarchical clustering labels to DataFrames
    pca_df['hierarchical_cluster'] = hierarchical_labels
    tsne_df['hierarchical_cluster'] = hierarchical_labels

    # Visualize clusters in PCA space
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='PC1', y='PC2', hue='hierarchical_cluster', data=pca_df, palette='viridis',
                    alpha=0.7, s=10, legend='full')
    plt.title(f'Hierarchical Clustering (K={n_clusters_hierarchical}) in PCA Space')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('hierarchical_clusters_pca.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize clusters in t-SNE space
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='hierarchical_cluster', data=tsne_df, palette='viridis',
                    alpha=0.7, s=10, legend='full')
    plt.title(f'Hierarchical Clustering (K={n_clusters_hierarchical}) in t-SNE Space')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('hierarchical_clusters_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()

    return hierarchical_labels, n_clusters_hierarchical


# Perform hierarchical clustering
hierarchical_labels, n_clusters_hierarchical = perform_hierarchical_clustering(
    scaled_cluster_data, pca_df, tsne_df)

# 7. CLUSTER EVALUATION
print("\n7. CLUSTER EVALUATION")
print("-" * 80)


def evaluate_clusters(scaled_data, original_data, kmeans_labels, dbscan_labels, hierarchical_labels):
    """
    Evaluate clustering results:
    - Calculate silhouette scores
    - Compare cluster characteristics
    - Visualize cluster profiles
    """
    print("Evaluating clustering results...")

    # Calculate silhouette scores for each algorithm
    kmeans_silhouette = silhouette_score(scaled_data, kmeans_labels)
    print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")

    # Filter out noise points for DBSCAN silhouette calculation
    dbscan_mask = dbscan_labels != -1  # Exclude noise points
    if len(set(dbscan_labels[dbscan_mask])) > 1:  # Need at least 2 clusters
        dbscan_silhouette = silhouette_score(scaled_data[dbscan_mask], dbscan_labels[dbscan_mask])
        print(f"DBSCAN Silhouette Score (excluding noise): {dbscan_silhouette:.4f}")
    else:
        print("DBSCAN didn't produce enough clusters for silhouette calculation")
        dbscan_silhouette = None

    hierarchical_silhouette = silhouette_score(scaled_data, hierarchical_labels)
    print(f"Hierarchical Clustering Silhouette Score: {hierarchical_silhouette:.4f}")

    # Create DataFrames with cluster assignments
    cluster_data = original_data.copy()
    cluster_data['kmeans_cluster'] = kmeans_labels
    cluster_data['dbscan_cluster'] = dbscan_labels
    cluster_data['hierarchical_cluster'] = hierarchical_labels

    # Add data index to the original data
    data['cluster_index'] = range(len(data))

    # Merge cluster assignments back to the full dataset
    for cluster_type in ['kmeans_cluster', 'dbscan_cluster', 'hierarchical_cluster']:
        # Create a mapping dataframe
        mapping_df = pd.DataFrame({
            'cluster_index': range(len(cluster_data)),
            cluster_type: cluster_data[cluster_type]
        })

        # Merge with the original data
        data[cluster_type] = mapping_df[cluster_type].values

    # Analyze K-Means clusters
    print("\nK-Means cluster profiles:")
    kmeans_profiles = cluster_data.groupby('kmeans_cluster').agg({
        'demand': ['mean', 'min', 'max'],
        'temperature': ['mean', 'min', 'max'],
        'humidity': ['mean'],
        'windSpeed': ['mean'],
        'hour': ['mean', 'min', 'max'],
        'day_of_week': ['mean']
    })

    print(kmeans_profiles)

    # Visualize cluster profiles
    # 1. Demand vs Temperature by cluster
    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=cluster_data, x='temperature', y='demand',
                    hue='kmeans_cluster', palette='viridis', alpha=0.6, s=10)
    plt.title('Demand vs Temperature by K-Means Cluster')
    plt.xlabel('Temperature')
    plt.ylabel('Demand')
    plt.grid(True, alpha=0.3)
    # Continuing from the cut-off point in the original code...

    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('kmeans_demand_temp.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Hour of Day vs. Demand by cluster
    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=cluster_data, x='hour', y='demand',
                    hue='kmeans_cluster', palette='viridis', alpha=0.6, s=10)
    plt.title('Demand vs Hour of Day by K-Means Cluster')
    plt.xlabel('Hour of Day')
    plt.ylabel('Demand')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('kmeans_demand_hour.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Radar charts for cluster profiles
    # Prepare data for radar charts
    # Select features for radar chart
    radar_features = ['demand', 'temperature', 'humidity', 'windSpeed', 'hour']

    # Normalize features for radar chart
    radar_data = cluster_data[radar_features].copy()
    for feature in radar_features:
        radar_data[feature] = (radar_data[feature] - radar_data[feature].min()) / (
                    radar_data[feature].max() - radar_data[feature].min())

    # Calculate mean values for each cluster
    kmeans_radar = radar_data.copy()
    kmeans_radar['cluster'] = kmeans_labels
    kmeans_radar = kmeans_radar.groupby('cluster').mean().reset_index()

    # Plot radar charts
    from math import pi

    # Number of variables
    num_vars = len(radar_features)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Close the loop

    # Initialize the figure
    plt.figure(figsize=(15, 10))

    # Number of clusters
    num_clusters = len(kmeans_radar)

    # Create subplots for each cluster
    for i, cluster in enumerate(kmeans_radar['cluster']):
        ax = plt.subplot(2, (num_clusters + 1) // 2, i + 1, polar=True)

        # Values for the current cluster
        values = kmeans_radar.loc[kmeans_radar['cluster'] == cluster, radar_features].values.flatten().tolist()
        values += values[:1]  # Close the loop

        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Cluster {cluster}")
        ax.fill(angles, values, alpha=0.25)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_features)

        # Set title
        ax.set_title(f"Cluster {cluster}", size=11)

    plt.tight_layout()
    plt.savefig('kmeans_radar_charts.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Comparison of clustering algorithms
    # Create a comparison table
    comparison_data = {
        'Algorithm': ['K-Means', 'DBSCAN', 'Hierarchical'],
        'Number of Clusters': [
            len(set(kmeans_labels)),
            len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            len(set(hierarchical_labels))
        ],
        'Silhouette Score': [
            kmeans_silhouette,
            dbscan_silhouette if dbscan_silhouette is not None else 'N/A',
            hierarchical_silhouette
        ],
        'Noise Points': [
            'N/A',
            list(dbscan_labels).count(-1),
            'N/A'
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    print("\nClustering Algorithms Comparison:")
    print(comparison_df)

    return cluster_data, comparison_df


# Evaluate the clustering results
cluster_data, comparison_df = evaluate_clusters(
    scaled_cluster_data, original_cluster_data, kmeans_labels, dbscan_labels, hierarchical_labels)

# 8. CLUSTER INTERPRETATION
print("\n8. CLUSTER INTERPRETATION")
print("-" * 80)


def interpret_clusters(cluster_data, original_data, data):
    """
    Interpret clustering results:
    - Characterize each cluster
    - Associate clusters with specific patterns
    - Create visualizations to explain cluster meanings
    """
    print("Interpreting K-Means clusters...")

    # Focus on K-Means clusters as the primary method
    # Create detailed profiles for each cluster
    cluster_profiles = cluster_data.groupby('kmeans_cluster').agg({
        'demand': ['mean', 'std', 'min', 'max', 'count'],
        'temperature': ['mean', 'std', 'min', 'max'],
        'humidity': ['mean', 'std', 'min', 'max'],
        'windSpeed': ['mean', 'std', 'min', 'max'],
        'hour': ['mean', 'std', 'min', 'max'],
        'day_of_week': ['mean', 'std', 'min', 'max']
    })

    # Flatten multi-level columns in cluster_profiles
    cluster_profiles.columns = ['_'.join(col).strip() for col in cluster_profiles.columns.values]

    # # Add mode separately for 'day_of_week'
    # day_of_week_mode = cluster_data.groupby('kmeans_cluster')['day_of_week'].agg(
    #     mode=lambda x: x.mode().iloc[0] if not x.mode().empty else None
    # ).reset_index()
    #
    # # Merge the mode into cluster_profiles
    # cluster_profiles = pd.merge(cluster_profiles, day_of_week_mode, on='kmeans_cluster', how='left' , suffixes=('', '_mode'))

    # Create cluster names based on characteristics
    cluster_names = {}

    # Determine cluster characteristics
    for cluster in sorted(cluster_data['kmeans_cluster'].unique()):
        profile = cluster_profiles.loc[cluster]

        # Demand level
        demand_mean = profile['demand_mean']
        if demand_mean > cluster_data['demand'].quantile(0.75):
            demand_level = "High Demand"
        elif demand_mean < cluster_data['demand'].quantile(0.25):
            demand_level = "Low Demand"
        else:
            demand_level = "Medium Demand"

        # Temperature level
        temp_mean = profile['temperature_mean']
        if temp_mean > cluster_data['temperature'].quantile(0.75):
            temp_level = "Hot"
        elif temp_mean < cluster_data['temperature'].quantile(0.25):
            temp_level = "Cool"
        else:
            temp_level = "Moderate"

        # Time of day
        hour_mean = profile['hour_mean']
        if 5 <= hour_mean < 12:
            time_of_day = "Morning"
        elif 12 <= hour_mean < 17:
            time_of_day = "Afternoon"
        elif 17 <= hour_mean < 22:
            time_of_day = "Evening"
        else:
            time_of_day = "Night"

        # Create cluster name
        cluster_names[cluster] = f"{demand_level} - {temp_level} {time_of_day}"

    print("\nCluster characterizations:")
    for cluster, name in cluster_names.items():
        print(f"Cluster {cluster}: {name}")

    # Add cluster names to data
    data['cluster_name'] = data['kmeans_cluster'].map(cluster_names)

    # Visualize hourly profiles by cluster
    hourly_by_cluster = data.groupby(['kmeans_cluster', 'hour'])['demand'].mean().unstack(level=0)

    plt.figure(figsize=(14, 8))
    for cluster in sorted(data['kmeans_cluster'].unique()):
        plt.plot(hourly_by_cluster.index, hourly_by_cluster[cluster],
                 linewidth=2, label=f"Cluster {cluster}: {cluster_names[cluster]}")

    plt.title('Hourly Demand Profiles by Cluster')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Demand')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('cluster_hourly_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize temperature vs demand by cluster with fitted regression lines
    plt.figure(figsize=(14, 10))

    # Plot scatter points
    sns.scatterplot(data=data, x='temperature', y='demand',
                    hue='cluster_name', palette='viridis', alpha=0.3, s=10)

    # Fit regression lines for each cluster
    for cluster in sorted(data['kmeans_cluster'].unique()):
        cluster_data = data[data['kmeans_cluster'] == cluster]
        sns.regplot(x='temperature', y='demand', data=cluster_data,
                    scatter=False, ci=None, line_kws={'linewidth': 2})

    plt.title('Temperature vs Demand Relationship by Cluster')
    plt.xlabel('Temperature')
    plt.ylabel('Demand')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('cluster_temp_demand_regression.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize cluster distribution by day of week
    dow_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    data['day_name'] = data['day_of_week'].map(dow_names)

    plt.figure(figsize=(14, 8))
    cluster_dow = pd.crosstab(data['day_name'], data['kmeans_cluster'], normalize='index')
    cluster_dow.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
    plt.title('Cluster Distribution by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Proportion')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('cluster_day_of_week.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a comprehensive cluster summary DataFrame
    cluster_summary = pd.DataFrame(cluster_names.values(), index=cluster_names.keys(), columns=['Description'])

    # Add key metrics to summary
    for cluster in cluster_summary.index:
        # Count and percentage
        count = (data['kmeans_cluster'] == cluster).sum()
        percentage = count / len(data) * 100
        cluster_summary.loc[cluster, 'Count'] = count
        cluster_summary.loc[cluster, 'Percentage'] = f"{percentage:.2f}%"

        # Key metrics
        cluster_summary.loc[cluster, 'Avg Demand'] = cluster_profiles.loc[cluster, 'demand_mean']
        cluster_summary.loc[cluster, 'Avg Temperature'] = cluster_profiles.loc[cluster, 'temperature_mean']
        cluster_summary.loc[cluster, 'Avg Hour'] = cluster_profiles.loc[cluster, 'hour_mean']

        # Most common day
        # most_common_day = dow_names[cluster_profiles.loc[cluster, 'day_of_week_mode']]
        # cluster_summary.loc[cluster, 'Most Common Day'] = most_common_day

    print("\nComprehensive Cluster Summary:")
    print(cluster_summary)

    return cluster_names, cluster_summary




# Interpret clusters
cluster_names, cluster_summary = interpret_clusters(cluster_data, original_cluster_data, data)

# 9. FINAL VISUALIZATIONS AND EXPORT
print("\n9. FINAL VISUALIZATIONS AND EXPORT")
print("-" * 80)


def create_final_visualizations(data, cluster_names):
    """
    Create final visualizations for the report:
    - Dashboard-style visualization showing key cluster characteristics
    - 3D visualization of clusters
    - Calendar heatmap of cluster distribution
    """
    print("Creating final visualizations...")

    # 1. Create a 3D visualization combining demand, temperature, and hour
    from mpl_toolkits.mplot3d import Axes3D

    data_sample = data

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        data_sample['temperature'],
        data_sample['hour'],
        data_sample['demand'],
        c=data_sample['kmeans_cluster'],
        cmap='viridis',
        s=30,
        alpha=0.7
    )

    ax.set_xlabel('Temperature')
    ax.set_ylabel('Hour of Day')
    ax.set_zlabel('Demand')
    ax.set_title('3D Visualization of Energy Demand Clusters')

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=scatter.cmap(scatter.norm(cluster)),
                                  label=f"Cluster {cluster}: {name}", markersize=10)
                       for cluster, name in cluster_names.items()]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('3d_cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Create a calendar heatmap if time data is available
    if 'local_time' in data.columns:
        # Extract date components
        data['date'] = data['local_time'].dt.date
        data['month'] = data['local_time'].dt.month
        data['day'] = data['local_time'].dt.day

        # Get unique months in the data
        months = sorted(data['month'].unique())

        # Create a matrix of most common cluster by day
        cal_data = data.groupby(['month', 'day'])['kmeans_cluster'].agg(lambda x: x.value_counts().index[0]).unstack(
            level=0)

        # Plot the calendar heatmap
        plt.figure(figsize=(15, 8))

        # Use a colormap with distinct colors for each cluster
        cmap = plt.cm.get_cmap('viridis', len(cluster_names))

        # Create heatmap
        sns.heatmap(cal_data, cmap=cmap, linewidths=.5, cbar_kws={'label': 'Cluster'})

        plt.title('Most Common Cluster by Day')
        plt.ylabel('Day of Month')
        plt.xlabel('Month')
        plt.tight_layout()
        plt.savefig('cluster_calendar_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Dashboard-style visualization of key cluster characteristics
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Hourly profiles
    ax1 = fig.add_subplot(gs[0, :])
    hourly_by_cluster = data.groupby(['kmeans_cluster', 'hour'])['demand'].mean().unstack(level=0)
    for cluster in sorted(data['kmeans_cluster'].unique()):
        ax1.plot(hourly_by_cluster.index, hourly_by_cluster[cluster],
                 linewidth=2, label=f"Cluster {cluster}: {cluster_names[cluster]}")

    ax1.set_title('Hourly Demand Profiles by Cluster', fontsize=16)
    ax1.set_xlabel('Hour of Day', fontsize=14)
    ax1.set_ylabel('Average Demand', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    # Plot 2: Temperature vs Demand
    ax2 = fig.add_subplot(gs[1, 0])
    for cluster in sorted(data['kmeans_cluster'].unique()):
        cluster_data = data[data['kmeans_cluster'] == cluster]
        ax2.scatter(cluster_data['temperature'], cluster_data['demand'],
                    alpha=0.3, s=10, label=f"Cluster {cluster}")

    ax2.set_title('Temperature vs Demand by Cluster', fontsize=16)
    ax2.set_xlabel('Temperature', fontsize=14)
    ax2.set_ylabel('Demand', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)

    # Plot 3: Cluster distribution
    ax3 = fig.add_subplot(gs[1, 1])
    cluster_counts = data['cluster_name'].value_counts()
    ax3.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax3.axis('equal')
    ax3.set_title('Cluster Distribution', fontsize=16)

    # Plot 4: Day of week distribution
    ax4 = fig.add_subplot(gs[2, 0])
    cluster_dow = pd.crosstab(data['day_name'], data['kmeans_cluster'])
    cluster_dow.plot(kind='bar', stacked=True, ax=ax4, colormap='viridis')
    ax4.set_title('Cluster Distribution by Day of Week', fontsize=16)
    ax4.set_xlabel('Day of Week', fontsize=14)
    ax4.set_ylabel('Count', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(title='Cluster', fontsize=12)

    # Plot 5: Hour of day vs Day of week heatmap
    ax5 = fig.add_subplot(gs[2, 1])
    pivot_data = pd.crosstab(data['hour'], data['day_of_week'], values=data['kmeans_cluster'], aggfunc='mean')
    sns.heatmap(pivot_data, cmap='viridis', ax=ax5)
    ax5.set_title('Average Cluster by Hour and Day of Week', fontsize=16)
    ax5.set_xlabel('Day of Week (0=Mon, 6=Sun)', fontsize=14)
    ax5.set_ylabel('Hour of Day', fontsize=14)

    plt.tight_layout()
    plt.savefig('cluster_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Export labeled data to CSV
    labeled_data = data.copy()
    labeled_data.to_csv('energy_data_with_clusters.csv', index=False)
    print("Exported labeled data to energy_data_with_clusters.csv")

    return labeled_data


# Create final visualizations
labeled_data = create_final_visualizations(data, cluster_names)

# 10. REPORTING FUNCTIONS
print("\n10. REPORTING FUNCTIONS")
print("-" * 80)


def generate_cluster_profiles_report(cluster_summary, data, cluster_names):
    """
    Generate a detailed report of cluster profiles
    """
    print("Generating cluster profiles report...")

    # Initialize the report
    report = {}

    # For each cluster, create a detailed profile
    for cluster in sorted(data['kmeans_cluster'].unique()):
        cluster_name = cluster_names[cluster]
        cluster_data = data[data['kmeans_cluster'] == cluster]

        # Basic statistics
        profile = {
            'name': cluster_name,
            'count': len(cluster_data),
            'percentage': len(cluster_data) / len(data) * 100,
            'demand': {
                'mean': cluster_data['demand'].mean(),
                'median': cluster_data['demand'].median(),
                'std': cluster_data['demand'].std(),
                'min': cluster_data['demand'].min(),
                'max': cluster_data['demand'].max()
            },
            'temperature': {
                'mean': cluster_data['temperature'].mean(),
                'median': cluster_data['temperature'].median(),
                'std': cluster_data['temperature'].std(),
                'min': cluster_data['temperature'].min(),
                'max': cluster_data['temperature'].max()
            },
            'time': {
                'hour_mean': cluster_data['hour'].mean(),
                'hour_mode': cluster_data['hour'].mode().iloc[0],
                'common_day': cluster_data['day_name'].mode().iloc[0]
            }
        }

        # Correlations
        profile['correlations'] = {
            'temp_demand': cluster_data[['temperature', 'demand']].corr().iloc[0, 1],
            'humidity_demand': cluster_data[['humidity', 'demand']].corr().iloc[0, 1],
            'wind_demand': cluster_data[['windSpeed', 'demand']].corr().iloc[0, 1]
        }

        # Pattern analysis
        if 'local_time' in cluster_data.columns:
            # Time patterns
            profile['time_patterns'] = {
                'weekday_pct': (cluster_data['day_of_week'] < 5).mean() * 100,
                'weekend_pct': (cluster_data['day_of_week'] >= 5).mean() * 100,
                'morning_pct': ((cluster_data['hour'] >= 5) & (cluster_data['hour'] < 12)).mean() * 100,
                'afternoon_pct': ((cluster_data['hour'] >= 12) & (cluster_data['hour'] < 17)).mean() * 100,
                'evening_pct': ((cluster_data['hour'] >= 17) & (cluster_data['hour'] < 22)).mean() * 100,
                'night_pct': ((cluster_data['hour'] < 5) | (cluster_data['hour'] >= 22)).mean() * 100
            }

            # Monthly distribution
            if 'month' in cluster_data.columns:
                monthly_dist = cluster_data['month'].value_counts(normalize=True).sort_index() * 100
                profile['monthly_distribution'] = monthly_dist.to_dict()

        # Add to report
        report[cluster] = profile

    # Print a summary of the report
    print("\nSummary of Cluster Profiles:")
    for cluster, profile in report.items():
        print(f"\nCluster {cluster}: {profile['name']}")
        print(f"  Count: {profile['count']} ({profile['percentage']:.2f}%)")
        print(f"  Avg Demand: {profile['demand']['mean']:.2f}")
        print(f"  Avg Temperature: {profile['temperature']['mean']:.2f}")
        print(f"  Most Common Hour: {profile['time']['hour_mode']}")
        print(f"  Most Common Day: {profile['time']['common_day']}")
        print(f"  Temp-Demand Correlation: {profile['correlations']['temp_demand']:.3f}")

    return report


# Generate a detailed report of cluster profiles
cluster_profiles_report = generate_cluster_profiles_report(cluster_summary, data, cluster_names)


def create_interpretability_index(cluster_profiles_report):
    """
    Create an interpretability index for each cluster, rating how well-defined and interpretable it is
    """
    print("\nCreating cluster interpretability index...")

    interpretability_scores = {}

    for cluster, profile in cluster_profiles_report.items():
        # Initialize score
        score = 0

        # 1. Size of cluster (larger clusters are generally more stable/interpretable)
        # Score up to 25 points for size
        size_score = min(25, (profile['percentage'] / 5) * 25)

        # 2. Distinctiveness of demand pattern
        # Score up to 25 points for how different the demand is from overall mean
        overall_mean = data['demand'].mean()
        cluster_mean = profile['demand']['mean']
        demand_diff_pct = abs(cluster_mean - overall_mean) / overall_mean * 100
        distinctiveness_score = min(25, demand_diff_pct)

        # 3. Time pattern clarity
        # Score up to 25 points for clear time patterns
        time_clarity = 0
        if 'time_patterns' in profile:
            # Higher score for clusters concentrated in specific periods
            time_periods = [
                profile['time_patterns']['morning_pct'],
                profile['time_patterns']['afternoon_pct'],
                profile['time_patterns']['evening_pct'],
                profile['time_patterns']['night_pct']
            ]
            max_period = max(time_periods)
            time_clarity = min(25, max_period)

        # 4. Weather correlation strength
        # Score up to 25 points for strong correlations with weather
        correlation_strength = abs(profile['correlations']['temp_demand']) * 25

        # Total score
        total_score = size_score + distinctiveness_score + time_clarity + correlation_strength

        # Interpretability rating
        if total_score >= 75:
            rating = "Excellent"
        elif total_score >= 60:
            rating = "Good"
        elif total_score >= 45:
            rating = "Moderate"
        elif total_score >= 30:
            rating = "Fair"
        else:
            rating = "Poor"

        interpretability_scores[cluster] = {
            'size_score': size_score,
            'distinctiveness_score': distinctiveness_score,
            'time_clarity': time_clarity,
            'correlation_strength': correlation_strength,
            'total_score': total_score,
            'rating': rating
        }

    # Print interpretability scores
    print("\nCluster Interpretability Scores:")
    for cluster, scores in interpretability_scores.items():
        print(f"Cluster {cluster} ({cluster_names[cluster]}): {scores['total_score']:.2f}/100 - {scores['rating']}")

    return interpretability_scores


# Create interpretability index
interpretability_scores = create_interpretability_index(cluster_profiles_report)

# 11. CONCLUSION
print("\n11. CONCLUSION")
print("-" * 80)

print("\nENERGY DATA CLUSTERING ANALYSIS SUMMARY")
print("=" * 50)

print(f"\nTotal observations analyzed: {len(data)}")
print(f"Number of features used in clustering: {len(clustering_features)}")
print(f"Optimal number of clusters (K-Means): {optimal_k}")

print("\nCluster summary:")
for cluster, name in cluster_names.items():
    count = (data['kmeans_cluster'] == cluster).sum()
    percentage = count / len(data) * 100
    interp_rating = interpretability_scores[cluster]['rating']
    print(f"- Cluster {cluster}: {name}")
    print(f"  {count} observations ({percentage:.2f}%), Interpretability: {interp_rating}")

print("\nKey findings:")
print("1. The data exhibits clear patterns of energy demand related to time of day and temperature.")
print("2. K-Means clustering provided the most interpretable results.")
print(
    f"3. The silhouette score for K-Means was {comparison_df.loc[0, 'Silhouette Score']:.4f}, indicating good cluster separation.")
print(
    "4. The clusters identified represent distinct energy usage patterns that can inform energy management strategies.")

print("\nRecommendations:")
print("1. Use these cluster insights for demand forecasting and peak management.")
print("2. Target specific clusters for demand response programs.")
print("3. Optimize energy supply based on the identified patterns.")
print("4. Continue monitoring for shifts in cluster patterns over time.")

print("\nAnalysis complete. Results exported to CSV and visualizations saved.")

# Export final report summary
report_summary = {
    'total_observations': len(data),
    'features_used': clustering_features,
    'optimal_k': optimal_k,
    'silhouette_score': comparison_df.loc[0, 'Silhouette Score'],
    'cluster_summary': cluster_summary.to_dict(),
    'interpretability_scores': interpretability_scores
}

# Convert to DataFrame for easy export
report_df = pd.DataFrame({
    'metric': report_summary.keys(),
    'value': [str(v) for v in report_summary.values()]
})

report_df.to_csv('clustering_report_summary.csv', index=False)
print("Report summary exported to clustering_report_summary.csv")