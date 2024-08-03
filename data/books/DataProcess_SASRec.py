from collections import defaultdict
from datetime import datetime
import json
import pandas as pd

f = open('./info/Books_5_2017-10-2018-11.txt', 'r')
items = f.readlines()
item_names = [_.split('\t')[0] for _ in items]
item_ids = [_.split('\t')[1] for _ in items]
item_dict = dict(zip(item_names, item_ids))

reviews = pd.read_csv('train_40000.csv')
output_file = open('train_40000_SASRec.txt', 'w')

import re

for idx in range(0, reviews.shape[0]):
    review = reviews.iloc[idx]
    titles = eval(review['history_item_title'])
    for title in titles:
        # print(title)
        # input()
        item_id = item_dict[title]
        output_file.write('%d %d\n' % (idx + 1, int(item_id)))
    title = review['item_title']
    item_id = item_dict[title]
    output_file.write('%d %d\n' % (idx + 1, int(item_id)))

output_file.close()
