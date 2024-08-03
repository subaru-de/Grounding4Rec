from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import os
import math
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# import argparse
# parse = argparse.ArgumentParser()
# parse.add_argument("--input_dir",type=str, default="./", help="your model directory")
# parse.add_argument("--input_dir",type=str, default="./", help="/data/guoxinsen/Grounding4Rec/Qwen1.5-1.8B")
# args = parse.parse_args()
# print(args.input_dir)

# path = []
# for root, dirs, files in os.walk(args.input_dir):
#     for name in files:
#             path.append(os.path.join(args.input_dir, name))
# print(path)

# f = open('./data/movie/movies.dat', 'r', encoding='ISO-8859-1')
# movies = f.readlines()
# movie_names = [_.split('::')[1].strip("\"") for _ in movies]
# movie_ids = [_ for _ in range(len(movie_names))]
# movie_dict = dict(zip(movie_names, movie_ids))

result_dict = dict()

for p in ['result_5000.json']:
    result_dict[p] = {
        "NDCG": [],
        "HR": [],
    }
    f = open(p, 'r')
    test_data = json.load(f)
    f.close()
    # text = [_["predict"][0].strip("\"") for _ in test_data]
    
    topk_list = [5, 10]
    NDCG = []
    for topk in topk_list:
        S = 0
        for i in range(len(test_data)):
            target_book = test_data[i]['output'].strip("\"")
            for j in range(topk):
                text = test_data[i]['predict'][j].strip("\"")
                if text == target_book:
                    S = S + (1 / math.log(j + 2))
                    break
            # rankId = rank[i][target_movie_id].item()
            # if rankId < topk:
            #     S = S + (1 / math.log(rankId + 2))
        NDCG.append(S / len(test_data) / (1 / math.log(2)))
    HR = []
    for topk in topk_list:
        S = 0
        for i in range(len(test_data)):
            target_book = test_data[i]['output'].strip("\"")
            for j in range(topk):
                text = test_data[i]['predict'][j].strip("\"")
                if text == target_book:
                    S = S + 1
                    break
            # rankId = rank[i][target_movie_id].item()
            # if rankId < topk:
            #     S = S + 1
        HR.append(S / len(test_data))
    print(NDCG)
    print(HR)
    print('_' * 100)
    result_dict[p]["NDCG"] = NDCG
    result_dict[p]["HR"] = HR

f = open('./evaluate_result.json', 'w')    
json.dump(result_dict, f, indent=4)