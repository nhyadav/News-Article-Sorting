# load he data
# perform tokenization
# perform text normalization (lemmatize)
# perform text representaion (tfidf)
# split the dataset in to training and testing data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from eda import get_parameter

def tokenization_normalization(data):
    wl = WordNetLemmatizer()
    combo = []
    for i in range(0,len(data)):
        reword = re.sub('[^a-zA-Z]',' ',data['Text'][i])
        reword = reword.lower()
        reword = reword.split()
        reword = [wl.lemmatize(word) for word in reword if word not in set(stopwords.words('english'))]
        reword = ' '.join(reword)
        combo.append(reword)
    return combo

def categorised_target(data, path):
    data['category_id'] = data['Category'].factorize()[0]
    data.to_csv(path)

def text_representation(t, x, y, px, py):
    X_train = t.fit_transform(x).toarray()
    X_test = t.transform(y).toarray()
    with open(px,'wb') as output:
        pickle.dump(X_train, output)
    with open(py,'wb') as output:
        pickle.dump(X_test, output)
    


def feature_engineering(params):
    ##################dataset_path######################
    train_path = params['data_source']['train']
    test_path = params['data_source']['test']
    sample_submission_path = params['data_source']['sample_submission']
    #####################tfidf parametrs################
    tfidf_param = params['tfidf']
    tfidf = TfidfVectorizer(tfidf_param)
    tfidf_path = params['load_data']['tfidf']
    with open(tfidf_path, 'wb') as output:
        pickle.dump(tfidf, output)
    # #####################save path after tfidf##########
    save_tfidf_train_path = params['load_data']['X_train']
    save_tfidf_test_path = params['load_data']['X_test']
    save_tfidf_ytrain_path = params['load_data']['y_train']
    save_tfidf_ytest_path = params['load_data']['y_test']
    # ##################save dataset after categoriesed###
    save_train = params['load_data']['train']
    save_test_true_data = params['load_data']['test_true']
    ####################import dataset##################
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    test_true_data = pd.read_csv(sample_submission_path)
    ####################categorised#####################
    categorised_target(train_data, save_train)
    categorised_target(test_true_data, save_test_true_data)
    # ######################Tokenization##################
    train_data_container = tokenization_normalization(train_data)
    test_data_container = tokenization_normalization(test_data)
    ####################Text representation#############
    text_representation(tfidf,
                        train_data_container,
                        test_data_container,
                        save_tfidf_train_path,
                        save_tfidf_test_path
                        )
    ################y_train,y_test#####################
    train_cat = params['train_category']
    test_cat = params['test_category']
    y_train = train_data.replace({'Category': train_cat})
    y_test = test_true_data.replace({'Category': test_cat})
    y_trn = y_train['Category'].values.reshape(-1, 1)
    y_tst = y_test['Category'].values.reshape(-1, 1)
    with open(save_tfidf_ytrain_path,'wb') as output:
        pickle.dump(y_trn, output)
    with open(save_tfidf_ytest_path,'wb') as output:
        pickle.dump(y_tst, output)
    




if __name__ == "__main__":
    path = "E:\\DataScience_internship_with_ineuron\\newsarticalesorting\\newsarticlesorting\\config\\params.yaml"
    params = get_parameter(path)
    feature_engineering(params)