import numpy as np
import matplotlib
import matplotlib.pylab as plt
from sklearn import preprocessing, decomposition
from matplotlib.lines import Line2D
def plot_PCA(x, y):
    """
    :param x: expression level of genes
    :param y: class of cancer
    """
    # Preprocessing and data preparation
    x = np.array(x)
    # Transform the 5 labels to numerical(0->4)
    preprocess = preprocessing.LabelEncoder()
    y = preprocess.fit_transform(y)
    # Decompose and fit to expression data
    pca = decomposition.PCA(n_components=5)
    component = pca.fit_transform(x)

    # Put labels on the axis
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    # Plot & Save
    colors = np.array(["blue", "green", "red", "yellow", "violet"])
    plt.scatter(component[:, 0], component[:, 1], c=colors[y])
    plt.title("PCA for cancer classes")
    custom_lines = [Line2D([0], [0], color="violet", lw=4),
                    Line2D([0], [0], color="blue", lw=4),
                    Line2D([0], [0], color="yellow", lw=4),
                    Line2D([0], [0], color="red", lw=4),
                    Line2D([0], [0], color="green", lw=4)]

    plt.legend(custom_lines, ['PRAD', 'BRCA', 'LUAD',"KIRC","COAD"])
    plt.savefig('./output/PCA/PCA.png')

