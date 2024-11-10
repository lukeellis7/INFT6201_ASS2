# clustering.py
# Clustering model functions for unsupervised learning
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd


def train_clustering_model(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return model, labels


def calculate_silhouette_score(X, labels):
    """
    Calculates the silhouette score for the given clustering labels.

    Parameters:
    - X (DataFrame): The dataset used for clustering.
    - labels (array): Cluster labels assigned to each data point.

    Returns:
    - score (float): The silhouette score for the clustering.
    """
    score = silhouette_score(X, labels)
    return score


def visualize_clusters(X, labels):
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Clustering Visualization')
    plt.show()


def kmeans_clustering_on_pca(pca_data, n_clusters=5):
    pca_numeric = pca_data.select_dtypes(include=[float, int])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(pca_numeric)
    pca_data['Cluster'] = clusters

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', data=pca_data, hue='Cluster', palette='viridis', alpha=0.7)
    plt.title(f'KMeans Clustering with {n_clusters} Clusters on PCA Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.show()

    return pca_data


def plot_elbow_method(X, max_clusters=10):
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal Clusters')
    plt.show()


def plot_silhouette_scores(X, max_clusters=10):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Optimal Clusters')
    plt.show()


def plot_davies_bouldin_scores(X, max_clusters=10):
    davies_bouldin_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_clusters + 1), davies_bouldin_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Scores for Optimal Clusters')
    plt.show()


def plot_labelled_clusters(pca_data, clustered_data, player_names, n_representative=3):
    """
    Plots the clusters with labels for representative players and ensures 'Lionel Messi' and 'Cristiano Ronaldo'
    are labelled if present in the dataset.

    Parameters:
    - pca_data (DataFrame): The PCA-transformed data with principal components.
    - clustered_data (DataFrame): The data with cluster assignments.
    - player_names (Series): The original player names for labelling.
    - n_representative (int): Number of random players to label per cluster.
    """

    # Ensure player names and cluster labels are in the PCA data
    pca_data = pca_data.copy()
    pca_data['Cluster'] = clustered_data['Cluster']
    pca_data['player_name'] = player_names

    # Filter out rows with NaN player names
    pca_data = pca_data.dropna(subset=['player_name'])

    # Plot the clusters with a scatterplot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_data, palette='viridis', alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KMeans Clustering with Labelled Players on PCA Data')

    # Loop through each cluster to label players
    for cluster_id in pca_data['Cluster'].unique():
        cluster_players = pca_data[pca_data['Cluster'] == cluster_id]
        selected_players = cluster_players.sample(n=min(n_representative, len(cluster_players)), random_state=42)

        for _, player in selected_players.iterrows():
            plt.text(player['PC1'], player['PC2'], player['player_name'], fontsize=8)

    # Always label 'Lionel Messi' and 'Cristiano Ronaldo' if they are in the dataset
    for name in ['Lionel Messi', 'Cristiano Ronaldo']:
        if name in pca_data['player_name'].values:
            player = pca_data[pca_data['player_name'] == name].iloc[0]
            plt.text(player['PC1'], player['PC2'], player['player_name'], fontsize=10, fontweight='bold', color='red')

    plt.legend(title="Cluster")
    plt.show()


def display_cluster_representatives(pca_data, clustered_data, player_names, top_n=30):
    """
    Displays the top `top_n` players in each cluster.

    Parameters:
    - pca_data (DataFrame): The PCA-transformed data with principal components.
    - clustered_data (DataFrame): The data with cluster assignments.
    - player_names (Series): The original player names for displaying.
    - top_n (int): Number of top players to display per cluster.
    """

    # Ensure player names and cluster labels are in the PCA data
    pca_data = pca_data.copy()
    pca_data['Cluster'] = clustered_data['Cluster']
    pca_data['player_name'] = player_names

    # Filter out rows with NaN player names
    pca_data = pca_data.dropna(subset=['player_name'])

    # Loop through each cluster and display the top `top_n` players
    for cluster_id in sorted(pca_data['Cluster'].unique()):
        print(f"\nCluster {cluster_id} Top {top_n} Players:")
        cluster_players = pca_data[pca_data['Cluster'] == cluster_id]
        top_players = cluster_players[['player_name', 'PC1', 'PC2']].head(top_n)
        print(top_players.to_string(index=False))


def display_cluster_summary(data, clusters, feature_columns, top_n=5):
    summary = []
    for cluster in clusters['Cluster'].unique():
        cluster_data = data[clusters['Cluster'] == cluster][feature_columns]
        summary.append(cluster_data.mean().to_frame(name=f"Cluster {cluster} Mean"))
    summary_df = pd.concat(summary, axis=1)
    print("Cluster Summary (Mean Characteristics for Each Cluster):")
    print(summary_df)
    return summary_df


def plot_labelled_clusters_with_summary(pca_data, clustered_data, labels, feature_columns=None, original_data=None):
    """
    Plots the clusters with 2D PCA, labeling representative players in each cluster,
    and calculates a summary of selected features for each cluster using original data.

    Parameters:
    - pca_data: DataFrame containing PCA-transformed data.
    - clustered_data: DataFrame with cluster labels.
    - labels: Series or list of player names to use as labels on the plot.
    - feature_columns: List of feature columns to summarize per cluster.
    - original_data: Original data DataFrame containing feature columns for summarization.

    Returns:
    - summary_df: DataFrame with the summary of features per cluster.
    """
    # Merge clustered data with labels and pca_data
    pca_data = pca_data.copy()
    pca_data['Cluster'] = clustered_data['Cluster']
    pca_data['player_name'] = labels  # Ensure 'player_name' column exists for labelling

    # Filter out rows with NaN player names to avoid 'nan' labels in the plot
    pca_data = pca_data.dropna(subset=['player_name'])

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_data, palette='viridis', alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA with Cluster Labels')

    # Loop through clusters to label representative players
    for cluster_id in pca_data['Cluster'].unique():
        cluster_players = pca_data[pca_data['Cluster'] == cluster_id]
        # Sample up to 3 players from each cluster, adjusting if fewer than 3 players are available
        sample_players = cluster_players.sample(n=min(3, len(cluster_players)), random_state=42)

        # Ensure specific players are labeled if they exist in the cluster
        for name in ["Lionel Messi", "Cristiano Ronaldo"]:
            if name in cluster_players['player_name'].values:
                sample_players = pd.concat([sample_players, cluster_players[cluster_players['player_name'] == name]])

        # Label selected players
        for _, player in sample_players.iterrows():
            plt.text(player['PC1'], player['PC2'], player['player_name'], fontsize=8)

    plt.legend(title="Cluster")
    plt.show()

    # Calculate and return summary statistics for selected feature columns per cluster using original data
    if feature_columns and original_data is not None:
        # Join clusters with original data to ensure feature columns are present
        clustered_data_with_features = clustered_data.join(original_data[feature_columns])
        summary_df = clustered_data_with_features.groupby('Cluster')[feature_columns].mean()
        return summary_df
    else:
        return None



