import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn import metrics



def logistic_regression(X, Y):
    """
	:param X: expression level of genes
	:param Y: class of cancer
    """

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=24)
    logreg = linear_model.LogisticRegression()
    logreg.fit(X_train, y_train)
    y_predict = logreg.predict(X_test)
    df_confusion = confusion_matrix(y_test, y_predict)
    cmap = plt.cm.RdPu
    plt.matshow(df_confusion, cmap=cmap)
    plt.title("Confusion Matrix/Heatmap Of Predicted Labels")
    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")
    plt.colorbar()
    plt.savefig("./output/Logistic Regression/logisticRegression.png")

    # Model accuracy for the training & test
    accuracy_training = metrics.accuracy_score(y_train, logreg.predict(X_train))
    print("Accuracy training:", accuracy_training)
    accuracy_test = metrics.accuracy_score(y_test, y_predict)
    print("Accuracy test:", accuracy_test)
    print("####")
    # Evaluation of the model i.e. 30% of data (test confusion matrix)
    confusion = pd.crosstab(y_predict, y_test)
    print(confusion)

