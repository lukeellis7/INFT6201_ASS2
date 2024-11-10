#feature_engineering.py
#Functions for creating and selecting features for modeling
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans

def create_performance_features(data):
    data['goal_xG_ratio'] = data['goals'] / data['xG']
    data['assist_xA_ratio'] = data['assists'] / data['xA']
    return data

def encode_categorical_features(data, columns):
    le = LabelEncoder()
    for column in columns:
        if column in data.columns:
            data[column] = le.fit_transform(data[column])
        else:
            print(f"Warning: '{column}' column not found in the dataset.")
    return data

def apply_pca(data, n_components=2, label_columns=None, n_neighbors=5):
    # Separate label columns if provided
    labels = data[label_columns] if label_columns else pd.DataFrame()

    # Select only numeric columns for PCA
    numeric_data = data.select_dtypes(include=[float, int])

    # Impute missing values using KNN imputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    numeric_data_imputed = imputer.fit_transform(numeric_data)

    # Standardize the data
    scaler = StandardScaler()
    numeric_data_standardized = scaler.fit_transform(numeric_data_imputed)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(numeric_data_standardized)
    pca_data = pd.DataFrame(pca_components, columns=[f'PC{i + 1}' for i in range(n_components)])

    # Concatenate label columns back with the PCA data for easy plotting
    pca_data = pd.concat([pca_data, labels.reset_index(drop=True)], axis=1)

    return pca_data, pca  # Return both pca_data and the PCA model (pca)

def kmeans_clustering_on_pca(pca_data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(pca_data.iloc[:, :pca_data.shape[1] - len(['teams_played_for', 'league'])])
    pca_data['Cluster'] = clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', data=pca_data, hue='Cluster', palette='viridis', alpha=0.7)
    plt.title(f'KMeans Clustering with {n_clusters} Clusters on PCA Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.show()
    return pca_data
