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




start:
python3 app.py

test
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

 result :
  - true - text doesn't contains a toxic comment
  - false - contains toxic comment

