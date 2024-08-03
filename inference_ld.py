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


if torch.cuda.is_available():
    device = "cuda:6"
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
    result_json_data: str = "result_5000.json",
    batch_size: int=8,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda:6":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            # device_map={'', 1},
        ).to(device)
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 1}
        ).to(device)
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

    # def custom_beam_search_with_constraints(model, tokenizer, input_ids, book_titles_tensor, beam_width=5, max_length=50):
    #     """
    #     自定义 beam search，并在每一步生成时仅允许生成书籍目录中的名称。
        
    #     Args:
    #         model: 预训练的语言模型。
    #         tokenizer: 对应的 tokenizer。
    #         input_ids: 输入的 token IDs。
    #         book_titles_tensor: 书籍目录中书籍名称的 token IDs。
    #         beam_width: beam search 的宽度。
    #         max_length: 生成序列的最大长度。
        
    #     Returns:
    #         List[str]: 解码后的书籍名称列表。
    #     """
    #     def get_valid_next_tokens(token_ids, book_titles_tensor):
    #         """
    #         获取当前 token_ids 可生成的有效下一个 token集合。
    #         """
    #         # 获取所有可能的下一个 token ID
    #         next_token_candidates = book_titles_tensor[:, len(token_ids):len(token_ids)+1]
    #         return next_token_candidates.unique()

    #     # 初始化 beam search 参数
    #     batch_size = input_ids.shape[0]
    #     beam_scores = torch.zeros((batch_size, beam_width), dtype=torch.float, device=input_ids.device)
    #     beam_sequences = input_ids.unsqueeze(1).repeat(1, beam_width, 1)  # shape: (batch_size, beam_width, seq_len)
        
    #     # 初始 token 的 logits
    #     outputs = model(input_ids=input_ids, output_hidden_states=True)
    #     next_token_logits = outputs.logits[:, -1, :]  # shape: (batch_size, vocab_size)

    #     # 获取有效的下一个 token
    #     valid_next_tokens = get_valid_next_tokens(input_ids[0], book_titles_tensor)
    #     next_token_logits[:, :] = float('-inf')
    #     next_token_logits[:, valid_next_tokens] = next_token_logits[:, valid_next_tokens]

    #     # 选择 top-k 的 token 作为初始 beam
    #     topk_scores, topk_tokens = torch.topk(next_token_logits, beam_width, dim=-1)
    #     beam_scores[:, :] = topk_scores
    #     beam_sequences[:, :, -1] = topk_tokens

    #     # 迭代生成 token
    #     for step in range(max_length - input_ids.shape[1]):
    #         all_scores = []
    #         all_sequences = []

    #         for beam_idx in range(beam_width):
    #             # 获取当前 beam 的序列
    #             current_sequences = beam_sequences[:, beam_idx, :]
    #             outputs = model(input_ids=current_sequences, output_hidden_states=True)
    #             next_token_logits = outputs.logits[:, -1, :]
                
    #             # 获取有效的下一个 token
    #             valid_next_tokens = get_valid_next_tokens(current_sequences[0], book_titles_tensor)
    #             next_token_logits[:, :] = float('-inf')
    #             next_token_logits[:, valid_next_tokens] = next_token_logits[:, valid_next_tokens]

    #             # 获取下一个 token 的分数
    #             next_token_scores = torch.log_softmax(next_token_logits, dim=-1)
                
    #             # 将当前 beam 的得分与下一个 token 的得分相加
    #             total_scores = beam_scores[:, beam_idx].unsqueeze(-1) + next_token_scores

    #             # 记录每个 beam 的序列和得分
    #             all_scores.append(total_scores)
    #             all_sequences.append(current_sequences.unsqueeze(2).repeat(1, 1, tokenizer.vocab_size))

    #         # 将所有 beam 的得分和序列拼接起来
    #         all_scores = torch.cat(all_scores, dim=-1)  # shape: (batch_size, beam_width * vocab_size)
    #         all_sequences = torch.cat(all_sequences, dim=2)  # shape: (batch_size, seq_len, beam_width * vocab_size)

    #         # 选择得分最高的 top-k beam
    #         topk_scores, topk_indices = torch.topk(all_scores, beam_width, dim=-1)
            
    #         # 更新 beam 的得分和序列
    #         beam_scores = topk_scores
    #         beam_indices = topk_indices // tokenizer.vocab_size
    #         next_tokens = topk_indices % tokenizer.vocab_size
    #         beam_sequences = torch.stack([all_sequences[b, :, beam_indices[b]] for b in range(input_ids.shape[0])], dim=0)
    #         beam_sequences[:, :, -1] = next_tokens

    #     # 获取最终生成的序列
    #     final_sequences = beam_sequences[:, 0, :]

    #     # 解码为可读文本
    #     decoded_sequences = tokenizer.batch_decode(final_sequences, skip_special_tokens=True)
    #     return decoded_sequences

    # # # 示例调用
    # # input_text = "Example input text"
    # # input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # # 假设 book_titles 是包含所有书籍名称的列表，并将其 tokenized
    # book_titles = ["Book Title 1", "Book Title 2", "Book Title 3"]
    # book_titles_tokenized = [tokenizer.encode(title, add_special_tokens=False) for title in book_titles]
    # book_titles_tensor = torch.tensor(book_titles_tokenized, dtype=torch.long)

    f = open('./data/books/info/Books_5_2017-10-2018-11.txt', 'r')
    books = f.readlines()
    item_names = [_.split('\t')[0] for _ in books]
    # item_ids = [_.split('\t')[1][:-1] for _ in books]
    # item_dict = dict(zip(item_ids, item_names))
    # result_dict = dict()
    # book_names = [_.split('::')[1].strip("\"") for _ in books]
    # book_ids = [_ for _ in range(len(book_names))]
    # book_dict = dict(zip(book_names, book_ids))

    item_name_ids = [tokenizer.encode('\"'+name+'\"', add_special_tokens=False) for name in item_names]
    # item_name_flattened_ids = list(set([token for sublist in item_name_ids for token in sublist]))
    # item_name_flattened_ids.append(tokenizer.eos_token_id)
    # item_name_flattened_ids.append(1)
    # item_name_flattened_ids.append(2)
    # print(item_name_ids)
    # ch = input()


    def evaluate(
        instructions,
        inputs=None,
        temperature=0,
        top_p=0.9,
        top_k=40,
        num_beams=10,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        # print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        # print(inputs.input_ids)
        count_input_tokens = len(inputs.input_ids[0])
        # print(count_input_tokens)

        # https://arxiv.org/abs/2010.00904
        def prefix_allowed_tokens_fn(batch_id, input_ids):
            # assert(len(input_ids) >= count_input_tokens)
            # print(tokenizer.batch_decode(input_ids[:], skip_special_tokens=False))

            generated_tokens = input_ids[count_input_tokens:].tolist()  # 获取当前生成的tokens
            # print('------------------------------')
            # print(generated_tokens)
            # print('--------------------')

            # output = tokenizer.batch_decode(input_ids[count_input_tokens:], skip_special_tokens=True)
            # print(output)
            # print('--------------------')


            # if len(generated_tokens) > 0:
            #     print(generated_tokens[-1], tokenizer.eos_token_id, ' qwq')

            # if len(generated_tokens) > 0 and generated_tokens[-1] == tokenizer.eos_token_id:
            #     print('qwq')
            #     input()
            #     return list([tokenizer.eos_token_id])

            valid_tokens = set()
            for name_ids in item_name_ids:
                if len(generated_tokens) == 0 or len(generated_tokens) < len(name_ids) and name_ids[:len(generated_tokens)] == generated_tokens:
                    valid_tokens.add(name_ids[len(generated_tokens)])
                elif len(generated_tokens) == len(name_ids) and name_ids[:len(generated_tokens)] == generated_tokens:
                    valid_tokens.add(tokenizer.eos_token_id)
            
            if not valid_tokens:
                valid_tokens.add(0)
            # print(list(valid_tokens))
            # assert(list(valid_tokens) is not None)
            return list(valid_tokens)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        s = generation_output.sequences
        # print(s)
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] for _ in output]
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        return real_outputs


    outputs = []
    from tqdm import tqdm
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
            output = evaluate(instructions, inputs)
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
