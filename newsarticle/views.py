from django.shortcuts import render
from django.http import request
import yaml
import pickle
import logging
import time
# Create your views here.

#########logging basic configuration
logging.basicConfig(filename="logging\\userlogging.txt",
                    filemode='a',
                    format='%(asctime)s %(levelname)s-%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG
                    )
logging.info('News Articles Sorting ready for use!')
# predict with model. 
path = "params.yaml"
try:
    with open(path, encoding='UTF-8') as data:
        parameteras = yaml.safe_load(data)
        logging.info("Paramters loaded successfully.")
    model_path = parameteras['logistic_regression']['save_model']
    tfidf_path = parameteras['load_data']['tfidf']
    train_code = parameteras['train_category']
    with open(model_path, 'rb') as data:
        model = pickle.load(data)
        logging.info("Model loaded successfully!.")
    with open(tfidf_path, 'rb') as data:
        tfidf = pickle.load(data)
        logging.info("tfidf vectorization loaded successfully!.")
except Exception as ex:
    logging.exception(ex)

def model_predict(text):
    try:
        
        x_test = tfidf.transform([text])
        logging.info("Text representation have done.")
        predict = model.predict(x_test)
        logging.info("Model has predicted successfully!.")
        probability = model.predict_proba(x_test)
        for category, id_ in train_code.items():    
            if id_ == predict[0]:
                predict_cateory = category
        return predict_cateory, probability
    except Exception as ex:
        logging.exception(ex)


def thresold_for_news(probability, result):
    final_category = result
    if probability >= 40:
        return final_category
    else:
        final_category = 'Others'
    return final_category

def home(request):
    return render(request, "newsarticle/index.html")

def predict(request):
    try:
        if request.method == 'POST':
            text = request.POST['text_data']
            start_time = time.time()
            if text:
                result, probability = model_predict(text)
                probability = round(probability.max()*100)
                news_category = thresold_for_news(probability, result)
                end_time = time.time()-start_time
                result_final = {'category': news_category, 'probability': probability, 'execution_time': end_time}
                logging.info("Output have printed on webpage.")
            else:
                result_final = {'category': "Please! put some text in textbox.", 'probability':0}
                logging.warning('Please put text in textbox.')
        else:
            result_final = {'category': "Something wrong!", 'probability':0}
            logging.error("post method in after classify button not working.")
    except Exception as ex:
        logging.exception(ex)
    return render(request, "newsarticle/index.html",result_final)




   
    