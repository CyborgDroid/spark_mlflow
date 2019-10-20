"""
Script to download the sample data files
"""

import requests, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.functions import GeneralMethods

data_urls = [
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names',
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/old.adult.names'
]

data_path = GeneralMethods.get_project_root() + '/data/'
os.makedirs(os.path.dirname(data_path), exist_ok=True)

for data in data_urls:
    r = requests.get(data)
    file_path =  data_path + data.split('/')[-1:][0]
    with open(file_path,'wb') as f:
        f.write(r.content)
    print('Saved :' +  file_path)