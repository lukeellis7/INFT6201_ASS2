# Ass2_Main.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from Include.data_preprocessing.data_cleaning import handle_missing_values, standardize_dates, remove_duplicates
from Include.data_preprocessing.feature_engineering import create_performance_features, encode_categorical_features, apply_pca
from Include.analysis.data_quality_check import identify_data_issues
from Include.analysis.exploratory_data_analysis import generate_summary_stats, visualize_distributions, correlation_analysis, plot_explained_variance, display_pca_loadings, plot_feature_distribution_per_cluster
from Include.models.supervised_classification import prepare_data_for_classification, train_svm_classifier, evaluate_svm_classifier, plot_confusion_matrix, plot_svm_feature_importance
from Include.models.clustering import kmeans_clustering_on_pca, plot_labelled_clusters_with_summary, display_cluster_representatives, plot_elbow_method, plot_silhouette_scores, plot_davies_bouldin_scores

# Step 1: Load datasets for all seasons and concatenate into a single DataFrame
seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020']
dataframes = [pd.read_csv(f'C:/Users/lukee/OneDrive/Desktop/{season}.csv') for season in seasons]
data = pd.concat(dataframes, ignore_index=True)

# Step 2: Perform data quality check to identify missing values and duplicates
print("Data Quality Issues:")
print(identify_data_issues(data))

# Step 3: Clean and preprocess data
# - Handle missing values using KNN imputer
# - Standardize dates in 'date' column (if it exists)
# - Remove duplicates based on player name and date
# - Create performance features and encode categorical variables!
data = handle_missing_values(data)
data = standardize_dates(data, 'date')
data = remove_duplicates(data, subset_columns=['player_name', 'date'] if 'date' in data.columns else ['player_name'])
data = create_performance_features(data)
data = encode_categorical_features(data, columns=['country', 'team'])

# Step 4: Perform exploratory data analysis
# - Display summary statistics for numerical columns
# - Plot distributions for key performance metrics
# Only display summary statistics for selected columns
selected_columns = ['goals', 'assists', 'minutes_played', 'key_passes']
print("Summary Statistics:")
print(generate_summary_stats(data[selected_columns]))
visualize_distributions(data, columns=['goals', 'assists', 'minutes_played', 'key_passes'])

# Step 5: Conduct correlation analysis to identify relationships between numerical features
correlation_matrix = correlation_analysis(data)

# Step 6: Apply PCA to reduce dimensionality and perform KMeans clustering
# - Apply PCA to get the top 3 principal components
# - Perform KMeans clustering with 5 clusters on PCA-transformed data
# - Plot clusters with representative player labels and display a summary table
pca_data, pca_model = apply_pca(data, n_components=3, label_columns=['teams_played_for', 'league'])

# Evaluate optimal clusters with different methods
X = pca_data.select_dtypes(include=[float, int])  # PCA-transformed data for evaluation

# Plot elbow method for WCSS
plot_elbow_method(X, max_clusters=10)

# Plot silhouette scores for different cluster counts
plot_silhouette_scores(X, max_clusters=10)

# Plot Davies-Bouldin scores for different cluster counts
plot_davies_bouldin_scores(X, max_clusters=10)

# Add PCA columns (PC1, PC2, PC3) to the main `data` DataFrame
data[['PC1', 'PC2', 'PC3']] = pca_data[['PC1', 'PC2', 'PC3']]

clustered_data = kmeans_clustering_on_pca(pca_data, n_clusters=5)
summary = plot_labelled_clusters_with_summary(pca_data, clustered_data, data['player_name'], feature_columns=['goals', 'assists', 'xG', 'xA'])

# Step 7: Display PCA loadings for insight into variable contributions to each principal component
feature_names = data.select_dtypes(include=[float, int]).columns
display_pca_loadings(pca_model, feature_names)

# Step 8: Map clusters to performance roles based on insights from the clustering analysis
performance_roles = {0: 'Low Impact', 1: 'Moderate Impact', 2: 'Role Player', 3: 'Key Contributor', 4: 'Top Performer'}
data['Performance_Role'] = clustered_data['Cluster'].map(performance_roles)

# Step 9: Prepare data for supervised classification
# - Define features as the top 3 principal components
# - Define target as the Performance_Role assigned from clusters
feature_columns = ['PC1', 'PC2', 'PC3']  # Use first 3 principal components
target_column = 'Performance_Role'
X_train, X_test, y_train, y_test = prepare_data_for_classification(data, feature_columns, target_column)

# Step 10: Train and evaluate SVM classifier on the prepared data
# - Train SVM classifier with radial basis function kernel
# - Evaluate model accuracy and display classification report
svm_model = train_svm_classifier(X_train, y_train, C=1.0, kernel='rbf', gamma='scale')
accuracy, classification_report = evaluate_svm_classifier(svm_model, X_test, y_test)
print("SVM Classifier Accuracy:", accuracy)
print("Classification Report:\n", classification_report)

# Step 11: Plot confusion matrix for the SVM model's predictions
plot_confusion_matrix(y_test, svm_model.predict(X_test))


# Step 12: Plot feature distributions for goals, assists, minutes_played, and key_passes per cluster
# Merge the cluster labels back into the original data DataFrame
data['Cluster'] = clustered_data['Cluster']  # Ensure the Cluster column is in the data DataFrame
# Now call the plotting function
features_to_plot = ['goals', 'assists', 'minutes_played', 'key_passes']
plot_feature_distribution_per_cluster(data, features_to_plot, cluster_column='Cluster')



# Step 13: Plot feature importance using SHAP to understand influence of each principal component
plot_svm_feature_importance(svm_model, X_train, use_kmeans=True, num_samples=100)


