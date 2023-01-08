import pandas as pd
import numpy as np
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def neural_network(X, Y):
    """
    :param X: expression level of genes
    :param Y: class of cancer
    """
    # Basic Neural network
    Y_encoded = []
    for cancer_class in Y:
        if cancer_class == "BRCA":
            Y_encoded.append(0)
        elif cancer_class == "PRAD":
            Y_encoded.append(1)
        elif cancer_class == "LUAD":
            Y_encoded.append(2)
        elif cancer_class == "KIRC":
            Y_encoded.append(3)
        elif cancer_class == "COAD":
            Y_encoded.append(4)
    y = to_categorical(Y_encoded)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=24, stratify=Y)
    init = 'random_uniform'
    # layers
    input_layer = Input(shape=(20531,))
    mid_layer = Dense(100, activation= 'relu', kernel_initializer=init)(input_layer)
    mid_layer2 = Dense(50, activation= 'relu', kernel_initializer=init)(mid_layer)
    ouput_layer = Dense(5, activation= 'softmax', kernel_initializer=init)(mid_layer2)
    model = Model(input_layer, ouput_layer)

    model.compile(optimizer='sgd', loss="binary_crossentropy", metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=100, verbose=1)

    y_predict = np.argmax(model.predict(x_test), axis=1)

    # model accuracy for test
    print("####")
    accuracy_test = accuracy_score(np.argmax(y_test, axis=1), y_predict)
    print("Accuracy test:", accuracy_test)

    # Evaluation of the model i.e. 30% of data (test confusion matrix)
    print("####")
    confusion = pd.crosstab(np.argmax(y_test, axis=1), y_predict)
    print(("BRCA = 0, PRAD=1, LUAD=2, KIRC=3, COAD=4"))
    print(confusion)
