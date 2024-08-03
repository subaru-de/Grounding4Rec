import sys

import fire
import gradio as gr
import torch
torch.set_num_threads(1)
import transformers
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig,  LlamaTokenizer
from transformers import LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

from accelerate import Accelerator
from accelerate.utils import set_seed


if torch.cuda.is_available():
    device = "cuda:4"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "Qwen/Qwen1.5-0.5B",
    lora_weights: str = "./Qwen1.5-0.5B_less_data/lora-alpaca",
    test_data_path: str = "data/books/test_5000.json",
    test_data_path_csv: str = "data/books/test_5000.csv",
    result_json_data: str = "result_D3_5000.json",
    batch_size: int=8,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda:4":
        # max_memory={0:"0GiB", 1:"10GiB", 2:"15GiB", 3:"0GiB", 4:"5GiB", 5:"0GiB", 6:"20GiB", 7:"20GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            # max_memory=max_memory,
            # device_map={'', 1},
        ).to(device)
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'', 6},
            # max_memory=max_memory,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    
    # model = model.to(device)
    
    # if torch.cuda.device_count() > 1:
    #     model = DataParallel(model)

    tokenizer.padding_side = "left"

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    f = open('./data/books/info/Books_5_2017-10-2018-11.txt', 'r')
    books = f.readlines()
    item_names = [_.split('\t')[0] for _ in books]
    item_ids = [int(_.split('\t')[1]) for _ in books]
    # item_names = item_names[:-2]; item_ids = item_ids[:-2] # get rid of this
    item_name_ids = [tokenizer.encode('\"'+name+'\"', add_special_tokens=False) for name in item_names]
    item_dict = dict(zip(item_names, item_ids))

    # text_free_output = torch.load('text_free_output.pt')
    
    from SASRec import SASRec
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=200, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default=device, type=str)
    parser.add_argument('--state_dict_path', default='../SASRec.pytorch/train_40000_SASRec_default/SASRec.epoch=1000.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth', type=str)

    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    print(len(item_names))
    args = parser.parse_args()
    text_free_model = SASRec(1, len(item_names), args).to(device)

    text_free_model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))

    seq = np.zeros([args.maxlen], dtype=np.int32)
    text_free_output = list()
    
    data = pd.read_csv(test_data_path_csv)
    count_test_data = data.shape[0]
    for idx in tqdm(range(count_test_data)):
        review = data.iloc[idx]
        titles = eval(review['history_item_title'])
        seq[args.maxlen - len(titles):] = np.array([item_dict[title] for title in titles])
        predictions = text_free_model.predict(*[np.array(l) for l in [[1], [seq], item_ids]])
        predictions = predictions[0]
        text_free_output.append(predictions)
    
    text_free_score = list()

    print("got text_free_output")

    class Trie:
        def __init__(self):
            self.cnt_node = 1
            self.next_node = [dict()]
            self.node_weight = [0.0]

        def insert(self, name_ids, weight):
            cur = 0
            self.node_weight[0] += weight
            
            for id in name_ids+[tokenizer.eos_token_id]:
                if self.next_node[cur].get(id) is None:
                    self.next_node[cur][id] = self.cnt_node
                    self.cnt_node += 1
                    self.next_node.append(dict())
                    self.node_weight.append(0)
                cur = self.next_node[cur].get(id)
                self.node_weight[cur] += weight
        
        def get_item_score(self, name_ids):
            item_score = list()
            cur = 0
            for id in name_ids+[tokenizer.eos_token_id]:
                nxt = self.next_node[cur][id]
                tmp_score = self.node_weight[nxt] / self.node_weight[cur]
                if (not tmp_score > 0) or math.isnan(tmp_score):
                    tmp_score = -math.inf
                else:
                    tmp_score = math.log(tmp_score)
                item_score.append(tmp_score)
                cur = nxt
            return item_score
    
    for i in range(count_test_data):
        print(i)
        cur_score = []
        item_trie = Trie()
        for name_ids, item_id in tqdm(zip(item_name_ids, item_ids)):
            item_trie.insert(name_ids, text_free_output[i][item_id])

        for name_ids in tqdm(item_name_ids):
            cur_score.append(item_trie.get_item_score(name_ids))
        
        text_free_score.append(cur_score)

    print("got text_free_score")

    def evaluate(
        batch_id,
        instructions,
        inputs=None,
        temperature=0,
        top_p=0.9,
        top_k=40,
        num_beams=10,
        max_new_tokens=128,
        **kwargs,
    ):
        print("batch: ", batch_id)
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # print("inputs.input_ids.shape: ", inputs.input_ids.shape)
        
        test_data_ids = torch.arange(inputs.input_ids.shape[0])
        inputs.input_ids[:, 0] = test_data_ids
        inputs.to(device)
        # inputs = Accelerator.prepare(inputs)

        
        count_input_tokens = len(inputs.input_ids[0])

        print("count_input_tokens: ", count_input_tokens)

        # https://arxiv.org/abs/2010.00904

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            generated_tokens = input_ids[count_input_tokens:].tolist()  # 生成的 tokens

            valid_tokens = set()
            for name_ids in item_name_ids:
                if len(generated_tokens) == 0 or \
                   len(generated_tokens) < len(name_ids) and name_ids[:len(generated_tokens)] == generated_tokens:
                    valid_tokens.add(name_ids[len(generated_tokens)])
                elif len(generated_tokens) == len(name_ids) and name_ids[:len(generated_tokens)] == generated_tokens:
                    valid_tokens.add(tokenizer.eos_token_id)
            
            if not valid_tokens:
                valid_tokens.add(0)
            
            return list(valid_tokens)

        from transformers import LogitsProcessor, LogitsProcessorList

        class CustomLogitsProcessor(LogitsProcessor):
            def __init__(self, weight_constant, item_name_ids, text_free_score):
                self.weight_constant = weight_constant
                self.item_name_ids = item_name_ids
                self.text_free_score = text_free_score
            
            # input_ids[0] should be test_data_id
            def __call__(self, input_ids, scores):
                # print(input_ids.shape)
                for beam in range(num_beams):
                    generated_tokens = input_ids[beam][count_input_tokens:].tolist()  # 获取当前生成的tokens
                    for name_ids, item_id in zip(item_name_ids, range(len(item_name_ids))):
                        if len(generated_tokens) == 0 or \
                           len(generated_tokens) < len(name_ids) and name_ids[:len(generated_tokens)] == generated_tokens:
                            pre_score = scores[beam][name_ids[len(generated_tokens)]]
                            # print(input_ids[0])
                            scores[beam][name_ids[len(generated_tokens)]] = self.weight_constant * pre_score + \
                                (1.0 - self.weight_constant) * self.text_free_score[input_ids[0][0]][item_id][len(generated_tokens)]
                        elif len(generated_tokens) == len(name_ids) and name_ids[:len(generated_tokens)] == generated_tokens:
                            pre_score = scores[beam][tokenizer.eos_token_id]
                            scores[beam][tokenizer.eos_token_id] = self.weight_constant * pre_score + \
                                (1.0 - self.weight_constant) * self.text_free_score[input_ids[0][0]][item_id][len(generated_tokens)]
                
                # print(scores.shape) [num_beams, num_all_tokens]
                return scores
            
        custom_processor = CustomLogitsProcessor(
            weight_constant=0.8,
            item_name_ids=item_name_ids,
            text_free_score=text_free_score[batch_size * batch_id: batch_size * (batch_id + 1)],
        )

        logits_processor = LogitsProcessorList([custom_processor])

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            length_penalty=0.0, # RLN in Debiasing-Divesifying Decoding methods
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                logits_processor=logits_processor,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        print("after generate")
        s = generation_output.sequences
        # print(s)
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] for _ in output]
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        return real_outputs


    outputs = []
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
        instructions = [_['instruction'] for _ in test_data]
        inputs = [_['input'] for _ in test_data]
        def batch(list, batch_size=batch_size):
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]
        for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs)))):
            instructions, inputs = batch
            output = evaluate(i, instructions, inputs)
            outputs = outputs + output
            
        for i, test in tqdm(enumerate(test_data)):
            test_data[i]['predict'] = outputs[i]


    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    fire.Fire(main)
