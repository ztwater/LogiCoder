from abc import ABC, abstractmethod
from concurrent.futures import as_completed, ProcessPoolExecutor
from itertools import chain
from openai import OpenAI
import os
import tiktoken
import torch
import tqdm
from unixcoder import UniXcoder
import voyageai

from globals import Globals
from myutils import FileUtils, PathUtils, Tokenizer


class Vectorizer(ABC):
    def __init__(self, input_file):
        self.input_file = input_file

    @abstractmethod
    def vectorize(self, content):
        pass

    @staticmethod
    @abstractmethod
    def get_name():
        pass

    def build(self):
        output_path = PathUtils.get_vector_path(self.input_file, self.get_name())
        if os.path.exists(output_path):
            print(f'Vectors for {os.path.basename(self.input_file)} have already been built.')
            return
        print(f'Build vectors for {os.path.basename(self.input_file)} using {self.get_name()}.')
        futures = dict()
        lines = FileUtils.load_pickle(self.input_file)
        with ProcessPoolExecutor(max_workers=(os.cpu_count())) as executor:  # program will get stuck if using all cpus
            for line in lines:
                futures[executor.submit(self.vectorize, line['content'])] = line

            new_lines = []
            t = tqdm.tqdm(total=len(futures))
            for future in as_completed(futures):
                line = futures[future]
                tokenized = future.result()
                new_lines.append({
                    'content': line['content'],
                    'metadata': line['metadata'],
                    'data': [{'embedding': tokenized}]
                })
                tqdm.tqdm.update(t)
            FileUtils.dump_pickle(new_lines, output_path)


class BagOfWordVectorizer(Vectorizer):
    def __init__(self, input_file, tokenizer):
        self.tokenizer = tokenizer
        super().__init__(input_file)

    def vectorize(self, content):
        return self.tokenizer.tokenize(content)

    @staticmethod
    def get_name():
        return 'bow'


class UniXCoderVectorizer(Vectorizer):
    def __init__(self, input_file, load_from_local=True):
        super().__init__(input_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if load_from_local:
            self.model = UniXcoder(Globals.UNIXCODER_PATH)
        else:
            self.model = UniXcoder("microsoft/unixcoder-base")
        self.model.to(self.device)

    def vectorize(self, content):
        tokens_ids = self.model.tokenize([content], max_length=512, mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(self.device)
        with torch.no_grad():
            tokens_embeddings, embedding = self.model(source_ids)
        return embedding

    @staticmethod
    def get_name():
        return 'unixcoder'

    def build(self, by_step=False, format='pkl'):
        output_path = PathUtils.get_vector_path(self.input_file, self.get_name())
        if os.path.exists(output_path):
            print(f'Vectors for {os.path.basename(self.input_file)} have already been built.')
            return
        print(f'Build vectors for {os.path.basename(self.input_file)} using {self.get_name()}.')
        if format == 'pkl':
            lines = FileUtils.load_pickle(self.input_file)
        else:
            lines = FileUtils.read_jsonl(self.input_file)
        new_lines = []
        t = tqdm.tqdm(total=len(lines))
        for line in lines:
            new_line = {
                'content': line['content'],
                'metadata': line['metadata'],
                'data': [dict()]
            }
            if by_step:
                embedding_by_step = [self.vectorize(step) for step in line['content']]
                new_line['data'][0]['embedding_by_step'] = embedding_by_step
            else:
                new_line['data'][0]['embedding'] = self.vectorize(line['content'])
            new_lines.append(new_line)
            tqdm.tqdm.update(t)
        FileUtils.dump_pickle(new_lines, output_path)


class TextEmbedding3Vectorizer(Vectorizer):
    def __init__(self, input_file, model_str='text-embedding-3-small'):
        super().__init__(input_file)
        self.model_str = model_str
        if model_str == "text-embedding-3-small":
            # self.url = "https://api.zhiyunai168.com/v1"
            # self.key = "sk-2fi8MRk6rldJovz7FTXZyYJigpGOR5srJ5cGaG2Lreb5mvnE"
            self.url = "https://api.kksj.org/v1"
            self.key = "sk-JIiLLdkQ7AWFlT8RA4A335330b78423895E32fC7A6FfFd4f"

    def vectorize(self, content):
        client = OpenAI(
            base_url=self.url,
            api_key=self.key  # os.environ.get("OPENAI_API_KEY")
        )
        content = content.replace("\n", " ")
        encoding = tiktoken.encoding_for_model("text-embedding-3-small")
        num_tokens = len(encoding.encode(content))
        print(f"Prompt token count: {num_tokens}")
        response = client.embeddings.create(
            input=[content],
            model=self.model_str
        )
        return response.data[0].embedding

    def build(self, by_step=False, format='pkl'):
        print(f'Build vectors for {os.path.basename(self.input_file)} using {self.get_name()}.')
        if format == 'pkl':
            lines = FileUtils.load_pickle(self.input_file)
        else:
            lines = FileUtils.read_jsonl(self.input_file)
        new_lines = []
        t = tqdm.tqdm(total=len(lines))
        for line in lines:
            new_line = {
                'content': line['content'],
                'metadata': line['metadata'],
                'data': [dict()]
            }
            if by_step:
                embedding_by_step = [self.vectorize(step) for step in line['content']]
                new_line['data'][0]['embedding_by_step'] = embedding_by_step
            else:
                new_line['data'][0]['embedding'] = self.vectorize(line['content'])
            new_lines.append(new_line)
            tqdm.tqdm.update(t)
        output_path = PathUtils.get_vector_path(self.input_file, self.get_name())
        FileUtils.dump_pickle(new_lines, output_path)

    @staticmethod
    def get_name():
        return 'text-embedding-3-small'

class VoyageVectorizer(Vectorizer):
    def __init__(self, input_file, input_type, for_code=False):
        super().__init__(input_file)
        self.input_type = input_type
        if for_code:
            self.model_str = 'voyage-code-3'
        else:
            self.model_str = 'voyage-3'
        self.tokenizer = Tokenizer()

    def vectorize(self, documents):
        client = voyageai.Client(api_key='pa--0bEJn0FrZzuhp4BRo-36Jl0Q2E2ygkLFGn0ZGGFDL0')
        if len(documents) == 0:
            return []
        non_empty_documents = []
        for doc in documents:
            if len(doc) == 0:
                non_empty_documents.append(' ')
            else:
                non_empty_documents.append(doc)
        try:
            embeddings = client.embed(non_empty_documents, model=self.model_str, input_type=self.input_type).embeddings
            return embeddings
        except Exception as e:
            print("Error:", e)
            if "Please lower the number of tokens in the batch." in str(e):
                batch_size = 16
                batch_num = len(non_empty_documents) // batch_size
                embeddings = []
                for i in range(batch_num):
                    start_line_no = i * batch_size
                    end_line_no = (i + 1) * batch_size
                    if i == batch_num - 1:
                        inputs = non_empty_documents[start_line_no:]  # include the last case
                    else:
                        inputs = non_empty_documents[start_line_no:end_line_no]
                    outputs = client.embed(inputs, model=self.model_str, input_type=self.input_type).embeddings
                    embeddings.extend(outputs)
                return embeddings
            else:
                print("Error Documents:\n", non_empty_documents)
                print([len(doc) for doc in non_empty_documents])
                raise Exception

    @staticmethod
    def get_name():
        return 'voyage'

    def build_by_batch(self, by_step=False, format='pkl'):
        print(f'Build vectors for {os.path.basename(self.input_file)} using {self.get_name()}.')
        output_path = PathUtils.get_vector_path(self.input_file, self.get_name())
        if os.path.exists(output_path):
            return
        if format == 'pkl':
            lines = FileUtils.load_pickle(self.input_file)
        else:
            lines = FileUtils.read_jsonl(self.input_file)
        contents = [line['content'] for line in lines]
        if by_step:
            group_lengths = [len(content) for content in contents]
            contents = list(chain.from_iterable(contents))
        embeddings = []
        batch_size = 128
        batch_num = len(contents) // batch_size + 1
        for i in tqdm.tqdm(range(batch_num), total=batch_num):
            start_line_no = i * batch_size
            end_line_no = (i + 1) * batch_size
            if i == batch_num - 1:
                inputs = contents[start_line_no:]  # include the last case
            else:
                inputs = contents[start_line_no:end_line_no]
            outputs = self.vectorize(inputs)
            embeddings.extend(outputs)
        if by_step:
            embeddings = self.group_strings_by_lengths(embeddings, group_lengths)
        new_lines = []
        for i, line in enumerate(lines):
            new_line = {
                'content': line['content'],
                'metadata': line['metadata'],
                'data': [dict()]
            }
            if by_step:
                new_line['data'][0]['embedding_by_step'] = embeddings[i]
            else:
                new_line['data'][0]['embedding'] = embeddings[i]
            new_lines.append(new_line)
        FileUtils.dump_pickle(new_lines, output_path)

    @staticmethod
    def group_strings_by_lengths(contents, group_lengths):
        result = []
        index = 0
        for length in group_lengths:
            # slice the list based on the current group length
            group = contents[index:index + length]
            result.append(group)
            index += length
        return result
