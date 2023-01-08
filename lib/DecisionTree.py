import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix

def decision_tree(X, Y):
    """
    :param X: expression level of genes
    :param Y: class of cancer
    """

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

    DTree = DecisionTreeClassifier()
    DTree = DTree.fit(X_train, y_train)

    plot_confusion_matrix(DTree, X_test, y_test, cmap=plt.cm.RdPu, normalize='true', display_labels=['BRCA', 'PRAD', 'COAD', 'LUAD', 'KIRC'])

    plt.title("Confusion matrix")
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.savefig("./output/Decision Tree/Decision_Tree.png")

    # model accuracy for the training & test
    accuracy_training = metrics.accuracy_score(y_train, DTree.predict(X_train))
    print("Accuracy training:", accuracy_training)
    accuracy_test = metrics.accuracy_score(y_test, DTree.predict(X_test))
    print("Accuracy testing : ", accuracy_test)



