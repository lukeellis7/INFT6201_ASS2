# supervised_classification.py
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import shap
import matplotlib.pyplot as plt



def prepare_data_for_classification(data, feature_columns, target_column):
    # Separate features and target
    X = data[feature_columns]
    y = data[target_column]

    # Impute missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Balance classes in the training set
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)

    return X_train_balanced, X_test, y_train_balanced, y_test


def balance_classes(X, y):
    data = X.copy()
    data['target'] = y
    majority_class = data['target'].value_counts().idxmax()
    majority_class_size = data['target'].value_counts().max()

    resampled_data = []

    for class_label, class_data in data.groupby('target'):
        if len(class_data) > 0:
            resampled_class = resample(class_data, replace=True, n_samples=majority_class_size, random_state=42)
            resampled_data.append(resampled_class)
        else:
            print(f"Skipping resampling for class '{class_label}' as it has no samples.")

    balanced_data = pd.concat(resampled_data)
    X_balanced = balanced_data.drop(columns=['target'])
    y_balanced = balanced_data['target']

    return X_balanced, y_balanced


def train_svm_classifier(X_train, y_train, C=1.0, kernel='rbf', gamma='scale'):
    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_svm_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return accuracy, report


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()


def plot_svm_feature_importance(model, X_train, use_kmeans=True, num_samples=100):
    """
    Plots the SHAP feature importance for the SVM model.

    Parameters:
    - model: Trained SVM model
    - X_train: Training data (features only)
    - use_kmeans: If True, uses shap.kmeans for background data summarization. Otherwise, shap.sample
    - num_samples: Number of samples to use for summarization if not using the entire dataset
    """
    # Summarize background data using K-means or random sampling
    if use_kmeans:
        background_data = shap.kmeans(X_train, num_samples)
    else:
        background_data = shap.sample(X_train, num_samples)

    # Create SHAP explainer with the summarized background data
    explainer = shap.KernelExplainer(model.predict, background_data)
    shap_values = explainer.shap_values(
        X_train[:num_samples])  # Limit SHAP values to a manageable subset for visualization

    # Plot feature importance
    shap.summary_plot(shap_values, X_train[:num_samples], plot_type="bar")

