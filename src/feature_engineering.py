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
from sklearn.model_selection import train_test_split
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

def categorised_target(data, path, train_ct):
    data['category_id'] = data['Category']
    data = data.replace({'category_id': train_ct})
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
    
    #####################tfidf parametrs################
    sublinear_tf = params['tfidf']['sublinear_tf']
    max_features = params['tfidf']['max_features']
    min_df = params['tfidf']['min_df']
    norm = params['tfidf']['norm']
    encoding = params['tfidf']['encoding']
    ngram_range = params['tfidf']['ngram_range']
    stop_words = params['tfidf']['stop_words']
    lowercase = params['tfidf']['lowercase']
    tfidf = TfidfVectorizer(
                            sublinear_tf = sublinear_tf,
                            max_features = max_features,
                            min_df = min_df,
                            norm = norm,
                            encoding = encoding,
                            ngram_range = (1, 2),
                            stop_words = stop_words,
                            lowercase = lowercase
    )
    tfidf_path = params['load_data']['tfidf']
    
    # #####################save path after tfidf##########
    save_tfidf_train_path = params['load_data']['X_train']
    save_tfidf_test_path = params['load_data']['X_test']
    save_tfidf_ytrain_path = params['load_data']['y_train']
    save_tfidf_ytest_path = params['load_data']['y_test']
    # ##################save dataset after categoriesed###
    save_train = params['load_data']['train']
    ####################import dataset##################
    train_data = pd.read_csv(train_path)
    
    ####################categorised#####################
    train_cat = params['train_category']
    categorised_target(train_data, save_train, train_cat)
    
    # ######################Tokenization##################
    # train_data_container = tokenization_normalization(train_data)
    train_data_container = tfidf.fit_transform(train_data.Text).toarray()
    with open(tfidf_path, 'wb') as output:
        pickle.dump(tfidf, output)
    #########################split data#################
    random_state = params['base']['random_state']
    y_trn = train_data.replace({'Category': train_cat})
    X_train,X_test,y_train,y_test = train_test_split(train_data_container,
                                                    y_trn['Category'],
                                                    test_size=0.20,
                                                    random_state=random_state)
    ####################Text representation#############
    with open(save_tfidf_train_path,'wb') as output:
        pickle.dump(X_train, output)
    with open(save_tfidf_test_path,'wb') as output:
        pickle.dump(X_test, output)
    ################y_train, y_test#####################
    with open(save_tfidf_ytrain_path,'wb') as output:
        pickle.dump(y_train, output)
    with open(save_tfidf_ytest_path,'wb') as output:
        pickle.dump(y_test, output)
    




if __name__ == "__main__":
    path = "E:\\DataScience_internship_with_ineuron\\newsarticalesorting\\newsarticlesorting\\params.yaml"
    params = get_parameter(path)
    feature_engineering(params)