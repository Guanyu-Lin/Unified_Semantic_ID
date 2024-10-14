
Code for the Paper ["**Unified Semantic and ID Representation Learning in Recommendation"**]

### requirements
```shell script
pip install -r requirements.txt
```

## data format
The data used in this paper can be downloaded via: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
```shell script
data preprocess
./data/data_process.py

generate negative items for testing
./data/generate_test.py


data-name.txt
one user per line
user_1 item_1 item_2 ...
user_2 item_1 item_2 ...

data-name_sample.txt
one user per line
user_1 neg_item_1 neg_item_2 ...
user_2 neg_item_1 neg_item_2 ...
```
To pre-compute the input item text into embedding:
```shell script
bash preprocess.sh 
```



## train
To train, validate and test the sequential recommendation model:
```shell script
bash run.sh 
```
