#Plotting functions (scatter plots, heatmaps, clustering visualizations)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.show()

def create_heatmap(data):
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Heatmap of Correlations')
    plt.show()
