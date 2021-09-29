# create a model for news classification
# train the model with training data.
# predict test data with model
# save the model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from eda import get_parameter
import json

def model_creation(X, y, p):
    c = p['C']
    c_w = p['class_weight']
    m_c = p['multi_class']
    pn = p['penalty']
    s = p['solver']
    lr_model = LogisticRegression(C = c,
                                class_weight = c_w,
                                multi_class = m_c,
                                penalty = pn,
                                solver = s)
    
    lr_model.fit(X, y)
    return lr_model

if __name__ == "__main__":
    path = "E:\\DataScience_internship_with_ineuron\\newsarticalesorting\\newsarticlesorting\\params.yaml"
    parameteras = get_parameter(path)
    params = parameteras['logistic_regression']['params']
    X_train_path = parameteras['load_data']['X_train']
    X_test_path = parameteras['load_data']['X_test']
    y_train_path = parameteras['load_data']['y_train']
    y_test_path = parameteras['load_data']['y_test']
    with open(y_train_path, 'rb') as data:
        y_train = pickle.load(data)
    with open(X_train_path, 'rb') as data:
        X_train = pickle.load(data)
    with open(X_test_path, 'rb') as data:
        X_test = pickle.load(data)
    with open(y_test_path, 'rb') as data:
        y_test = pickle.load(data)
    #########################model creation###############
    model = model_creation(X_train, y_train, params)
    m_params = parameteras['reports']['parameters']
    m_score = parameteras['reports']['scores']
    
    #######################save the model#################
    save_model_path = parameteras['logistic_regression']['save_model']
    with open(save_model_path, 'wb') as data:
        pickle.dump(model, data)
    ######################################################
    test_predict_data = model.predict(X_test)
    x_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, x_pred)
    test_accuracy = accuracy_score(y_test, test_predict_data)
    classification_report = classification_report(y_test, test_predict_data)
    confusion_matrix = confusion_matrix(y_test, test_predict_data)
    print("training accuracy:", train_accuracy)
    print("testing accuracy:",test_accuracy)
    
    ###########################save the score and parameters for model############
    with open(m_params, 'w') as data:
        param = {
                "C": params['C'],
                "class_weight": params['class_weight'],
                "multi_class": params['multi_class'],
                "penalty": params['penalty'],
                "solver": params['solver']
                }
        json.dump(param, data, indent=4)
    with open(m_score, 'w') as data:
        score = {
            "Training Accuracy:": train_accuracy,
            "Testing Accuracy:": test_accuracy,
        }
        json.dump(score, data, indent=4)