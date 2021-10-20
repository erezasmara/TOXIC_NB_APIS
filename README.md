# Toxic Comments Classification Naive base
Project Structure:
```
Toxic_nb_apis
├─ .DS_Store
├─ Procfile
├─ README.md
├─ app.py
├─ requirements.text
├─ test.csv
└─ train.csv

```
Description:
api model that’s capable of detecting if comment is a toxic. base on naive base models

Requirement:
- Python 3.8 or above
- pip module
- pandas module
- sklearn module
- data set train.csv & test.csv


Installation with pip
```
## sudo python3 -m pip install -U pandas
## sudo python3 -m pip install -U scikit-learn
## sudo python3 -m pip install -U termcolor
```

CURL
```
curl -X POST -H "Content-Type: application/json" -d '{"data":"HELLO"}' http://localhost:5000
```
response
```
{
  "msg": "have a nice day", 
  "result": true
}
```

- response.result :
  - true - text doesn't contains a toxic comment
  - false - contains toxic comment



CODE
```py
from flask import Flask,request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB



#####################################
'''this project made by: Erez Asmara '''
#####################################

######################################################################
#Load all the csv files
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
######################################################################


#######################################################################
#Split the data & transform
d = train
split = 0.7
d_train = d[:int(split*len(d))]
d_test = d[int((1-split)*len(d)):]
vectorizer = CountVectorizer(lowercase=False)
features = vectorizer.fit_transform(d_train.comment_text)
test_features = vectorizer.transform(d_test.comment_text)
######################################################################



######################################################################
#BUILD NAÏVE BAYES CLASSIFIER MODEL
model1 = MultinomialNB()
model1.fit(features, d_train.toxic)



######################################################################
#Build a predict function
def Predict_Func(string):
 answer=model1.predict(vectorizer.transform([string]))[0]
 if(answer !=1 ):
  return "OK"
 else:
    return "ERROR"
######################################################################



## flask server
app = Flask(__name__)
@app.route('/',methods = ["POST"])
def predict():
    data = request.get_json(force=True)
    message = data["data"]
    output = {'msg':message,'result': Predict_Func(message.lower())}
    
    return output

if __name__ == '__main__':
    app.run(port=5000,debug=True)


```
