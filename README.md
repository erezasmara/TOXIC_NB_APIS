# Toxic Comments Classification Naive base API
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
this api service use machine learning (naive base) to detect if your text is toxic ,
the server that handle the request done with flask.

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




Start:\
python3 app.py

testing with CURL
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

 result :
  - true - text doesn't contains a toxic comment
  - false - contains toxic comment

