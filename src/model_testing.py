import joblib
import re
import pandas as pd
import json
from eda import get_parameter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from  feature_engineering import tokenization_normalization
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


def get_category_name(category_id, category_codes):
    for category, id_ in category_codes.items():    
        if id_ == category_id:
            return category



def model_testing(parameteras):
    model_path = parameteras['logistic_regression']['save_model']
    test_report_path = parameteras['reports']['test_report']
    ####################load model##################################
    model = joblib.load(model_path)
    train_cat = parameteras['train_category']
    tfidf_path = parameteras['load_data']['tfidf']
    ###################load tfidf###################################
    tfidf = joblib.load(tfidf_path)
    ##################predict news text###################################
    #####################load the testcase##########################
    
    true_predict_class = parameteras['test_case_true_class']
    test_report = {}
    with open(test_report_path, 'w') as data:
        for i in range(1,8):
            test_data = parameteras['test_case']['case'+str(i)] 
            X_test = tfidf.transform([test_data])
            predict_score = model.predict(X_test)
            predict_prob = model.predict_proba(X_test)[0]
            prediction = get_category_name(predict_score, train_cat)
            probability = predict_prob.max()*100
            print("Prabability of News predicting:",prediction)
            print("The conditional probability is: %a" %probability)
            case = {
            'case'+str(i): {'true_class': true_predict_class[i-1],
                            'predict_class': prediction,
                            'predict_probability': probability
                            }
            }
            test_report.update(case)
        json.dump(test_report, data, indent=4)

if __name__ == "__main__":
    path = "params.yaml"
    parameteras = get_parameter(path)
    model_testing(parameteras)
   
