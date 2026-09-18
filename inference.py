 import argparse
from concurrent.futures import as_completed, ProcessPoolExecutor
import json

import openai
from openai import OpenAI, RateLimitError
import os
import tiktoken
import time
import tqdm

from globals import Globals
from myutils import FileUtils

class Model:
    def __init__(self, model_str, url, api_key, T, mode):
        self.model_str = model_str
        self.url = url
        self.api_key = api_key
        self.T = T
        self.max_retries = 3
        self.mode = mode

    def send_request(self, message_history):
        client = OpenAI(
            base_url=self.url,
            api_key=self.api_key
        )

        prompt_string = ''.join([m['content'] for m in message_history if m['role'] != 'system'])
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens = len(encoding.encode(prompt_string))
        print(f"Prompt token count: {num_tokens}")

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                if self.model_str == 'kimi-k2.6':
                    response = client.chat.completions.create(
                        model=self.model_str,
                        messages=message_history,
                        extra_body={
                            "thinking": {"type": "disabled"}
                        },
                        max_tokens=2048
                    )
                elif self.model_str == 'deepseek-v4-pro':
                    response = client.chat.completions.create(
                        model=self.model_str,
                        messages=message_history,
                        temperature=self.T,
                        extra_body={
                            "thinking": {"type": "disabled"}
                        },
                        max_tokens=2048
                    )
                else:
                    response = client.chat.completions.create(
                        model=self.model_str,
                        messages=message_history,
                        temperature=self.T,
                        max_tokens=2048
                    )
                break
            except RateLimitError as e:
                self.handle_gpt_rate_limit(e)
                retry_count += 1
            except openai.BadRequestError as e:
                print(e)
                if e.code == 'context_length_exceeded':
                    return ''
                retry_count += 1
            except Exception as e:
                print(e)
                # print(e.__dict__)
                # print(message_history[-1]['content'])
                retry_count += 1
            print(f"Retry count: {retry_count}")
        else:
            raise Exception("Unknown exception exists, please check the output.")
        return response.choices[0].message.content

    def run_prompts(self, prompts, system_prompt=None):
        if not system_prompt:
            system_prompt = "You are a helpful assistant."
        message_history = [{"role": "system", "content": system_prompt}]
        for prompt in prompts:
            message_history.append({"role": "user", "content": prompt})
            output = self.send_request(message_history)
            message_history.append({"role": "assistant", "content": output})
        return message_history

    def run_task(self, task_item):
        worker_idx, prompt_list, output_path, N = task_item
        # get finished completions
        if os.path.exists(output_path):
            results = read_jsonl(output_path)
            finished = [self.get_result_namespace(r) for r in results]
            output_file_handler = open(output_path, 'a')
        else:
            finished = []
            output_file_handler = open(output_path, 'w')
        print(f'Worker {worker_idx} starts, finished {len(finished)}/{len(prompt_list)}.')
        for prompt_line in tqdm.tqdm(prompt_list, total=len(prompt_list), desc=f'Worker {worker_idx}'):
            system_prompt = prompt_line['system_prompt']
            instruction_prompt = prompt_line['instruction_prompt']
            namespace = self.get_prompt_namespace(prompt_line)
            if namespace in finished:
                continue
            predictions = []
            for turn_id in range(N):
                # if prompt has system prompt, using it to initialize the llm
                message_history = self.run_prompts([instruction_prompt], system_prompt)
                output = message_history[-1]["content"]
                predictions.append(output)
            prediction_line = self.make_prediction_line(predictions, prompt_line)
            output_file_handler.write(json.dumps(prediction_line) + '\n')
            output_file_handler.flush()

    def get_result_namespace(self, line):
        if self.mode == 'doc':
            return line['namespace']
        elif self.mode == 'judge':
            return line['index']
        else:
            return line['metadata']['namespace']

    def get_prompt_namespace(self, line):
        if self.mode == 'rag':
            return line['query_window']['metadata']['namespace']
        else:
            return line['metadata']['namespace']

    def make_prediction_line(self, predictions, line):
        if self.mode == 'rag':
            return {
                'prompt': line['instruction_prompt'],
                'metadata': line['query_window']['metadata'],
                'predictions': [{'text': pred} for pred in predictions]
            }
        else:
            return {
                'prompt': line['instruction_prompt'],
                'metadata': line['metadata'],
                'predictions': [{'text': pred} for pred in predictions]
            }

    def handle_gpt_rate_limit(self, e):
        if 'requests per day' in str(e):
            print(f"Daily rate limit error with {self.api_key}.")
        elif 'requests per min' in str(e) or 'tokens per min' in str(e):
            print(f"Minute rate limit error with {self.api_key}, wait 60s.")
            time.sleep(60)
        elif 'exceeded your current quota' in str(e):
            print(f"Quota exceeded with {self.api_key}.")
        else:
            print("Unknown rate limit error.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_str', type=str, required=True)
    parser.add_argument('--T', type=float, default=0)
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--mode', type=str, required=True)
    return parser.parse_args()


def get_api_keys(model_str, max_parallel=12):
    if model_str in ["model_str"]:
        url = "https://online_api_url/v1"
        api_keys = FileUtils.read_file_as_list(Globals.API_KEY_FILE)[:max_parallel]
    else:
        url = "http://localhost:1234/v1"
        api_keys = ['example_api_key']
    return url, api_keys


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            js = json.loads(line)
            data.append(js)
    return data


def dump_jsonl(obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf8') as f:
        for item in obj:
            f.write(json.dumps(item) + '\n')


def get_output_path(args):
    sub_dir = os.path.basename(os.path.dirname(args.prompt_path))
    prompt_name = os.path.basename(args.prompt_path).split('.')[0]
    if args.T == 0:
        return os.path.join(args.output_dir, f'completion_{sub_dir}_{prompt_name}_{args.model_str}_T0_N{args.N}.jsonl')
    return os.path.join(args.output_dir, f'completion_{sub_dir}_{prompt_name}_{args.model_str}_T{args.T}_N{args.N}.jsonl')


if __name__ == "__main__":
    args = parse_args()
    print(args)
    url, api_keys = get_api_keys(args.model_str)
    num_of_apis = len(api_keys)
    prompts = read_jsonl(args.prompt_path)
    batch_len = len(prompts) // num_of_apis
    runners = []
    for api_idx in range(num_of_apis):
        start_line_no = api_idx * batch_len
        end_line_no = (api_idx + 1) * batch_len
        if api_idx == num_of_apis - 1:
            prompt_list = prompts[start_line_no:]  # include the last case
        else:
            prompt_list = prompts[start_line_no:end_line_no]
        api_key = api_keys[api_idx]
        model = Model(args.model_str, url, api_key, args.T, args.mode)
        output_path = get_output_path(args).replace('.jsonl', f'_batch{api_idx}.jsonl')
        task_item = api_idx, prompt_list, output_path, args.N
        runners.append((model, task_item))

    with ProcessPoolExecutor(max_workers=num_of_apis) as executor:
        futures = {executor.submit(model.run_task, task_item) for model, task_item in runners}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            future.result()

    merged = []
    for file_name in os.listdir(args.output_dir):
        file_path = os.path.join(args.output_dir, file_name)

        batch_prefix = get_output_path(args)[:-6]  # remove '.jsonl' extension
        if not os.path.isdir(file_path) and file_path.startswith(batch_prefix):
            batch = read_jsonl(file_path)
            merged.extend(batch)
            os.remove(file_path)
    dump_jsonl(merged, get_output_path(args))
