import requests

# with the model being served with mflow serve model, or the dockerized model running:

data = {"columns":["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"],
    "index":[0,1],
    "data":[[25,"Private",226802,"11th",7,"Never-married","Machine-op-inspct","Own-child","Black","Male",0.0,0.0,40.0,"United-States"],[38,"Private",89814,"HS-grad",9,"Married-civ-spouse","Farming-fishing","Husband","White","Male",0.0,0.0,50.0,"United-States"]]}

r = requests.post('http://127.0.0.1:5150/invocations', )
curl  -H 'Content-Type: application/json' -d 