#exploratory_data_analysis.py
#Functions for generating initial data summaries, visualizations, and insights
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_summary_stats(data):
    return data.describe()

def visualize_distributions(data, columns):
    for column in columns:
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()

def correlation_analysis(data):
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of Numeric Metrics")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.show()
    return corr_matrix

def plot_explained_variance(data, max_components=10):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    numeric_data = data.select_dtypes(include=[float, int])
    imputer = SimpleImputer(strategy='mean')
    numeric_data_imputed = imputer.fit_transform(numeric_data)

    pca = PCA(n_components=min(max_components, numeric_data_imputed.shape[1]))
    pca.fit(numeric_data_imputed)
    cumulative_variance = pca.explained_variance_ratio_.cumsum()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Number of Principal Components')
    plt.grid()
    plt.show()


def display_pca_loadings(pca_model, feature_names):
    # Verify that feature_names match the shape of PCA components
    if len(feature_names) != pca_model.components_.shape[1]:
        feature_names = feature_names[:pca_model.components_.shape[1]]
        print("Adjusted feature names to match PCA components' shape.")

    loadings = pca_model.components_.T * np.sqrt(pca_model.explained_variance_)
    loadings_df = pd.DataFrame(loadings, columns=[f'PC{i + 1}' for i in range(len(pca_model.components_))],
                               index=feature_names)
    print("PCA Loadings (Contribution of each feature to each principal component):")
    print(loadings_df)
    return loadings_df


def plot_feature_distribution_per_cluster(data, features, cluster_column='Cluster'):
    """
    Plots the distribution of specified features for each cluster without x and y axis limits.

    Parameters:
    - data: DataFrame containing the dataset.
    - features: List of features to plot.
    - cluster_column: Column in the data that contains cluster labels.
    """
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=data,
            x=feature,
            hue=cluster_column,
            multiple="stack",
            kde=True,
            palette="viridis",
            legend=True
        )

        plt.yscale('linear')  # Set y-axis to linear scale
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {feature} per Cluster")

        # Use plt.legend() only once, after plotting data with labels
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(handles=handles, labels=labels, title=cluster_column)

        plt.show()





