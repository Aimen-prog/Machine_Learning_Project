import pandas as pd
from lib.pca import plot_PCA
from lib.LogisticRegression import logistic_regression
from lib.DecisionTree import decision_tree
from lib.NeuralNetwork import neural_network

# Gene expressions data
df_data = pd.read_csv('./data/data.csv', header=None)

# Type of cancers (labels)
df_labels = pd.read_csv('./data/labels.csv', header=None)
X = df_data.iloc[1:, 1:]
Y = df_labels.iloc[1:, 1]


if __name__ == "__main__":
    # STAT
    # Principal Component Analysis
    plot_PCA(X,Y)

    # ML
    # Logistic regression
    logistic_regression(X,Y)
    # Decision tree
    decision_tree(X,Y)
    # ANN
    # Re-setting the data
    df_data = pd.read_csv("./data/data.csv", header=1)
    df_label = pd.read_csv("./data/labels.csv", header=1)
    Y = df_label.iloc[:,1]
    X = df_data.iloc[:,1:]
    neural_network(X,Y)