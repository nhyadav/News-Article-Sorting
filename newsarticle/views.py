from django.shortcuts import render
from django.http import request
import yaml
import pickle
# Create your views here.



# predict with model. 
path = "E:\\DataScience_internship_with_ineuron\\newsarticalesorting\\newsarticlesorting\\params.yaml"
with open(path) as data:
    parameteras = yaml.safe_load(data)

def model_predict(text):
    model_path = parameteras['logistic_regression']['save_model']
    tfidf_path = parameteras['load_data']['tfidf']
    train_code = parameteras['train_category']
    with open(model_path, 'rb') as data:
        model = pickle.load(data)
    with open(tfidf_path, 'rb') as data:
        tfidf = pickle.load(data)
    x_test = tfidf.transform([text])
    predict = model.predict(x_test)
    probability = model.predict_proba(x_test)
    for category, id_ in train_code.items():    
        if id_ == predict[0]:
            predict_cateory = category
    return predict_cateory, probability
    

def home(request):
    return render(request, "newsarticle/index.html")

def predict(request):
    if request.method == 'POST':
        text = request.POST['text_data']
        result, probability = model_predict(text)
        probability = round(probability.max()*100)
        result_final = {'category':result, 'probability': probability}
        print(result)
        print(probability)
    return render(request, "newsarticle/index.html",result_final)




   
    