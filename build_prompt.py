import os
import textwrap
from typing import Optional

from globals import Globals
from myutils import DataUtils, FileUtils, Tokenizer
from templates import (COT_PROMPT_TEMPLATE, COT_AND_CODE_PROMPT_TEMPLATE, SELF_PLANNING_PROMPT_TEMPLATE,
                       RAG_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE, LONG_CONTEXT_PROMPT_TEMPLATE)

class BasePromptBuilder:
    def __init__(self, bench='deveval'):
        self.bench = bench
        self.repo_base_dir = Globals.REPO_BASE_DIR if bench == 'deveval' else Globals.REPO_BASE_CODEREVAL_DIR

    def build_context_above(self, metadata, context_length: Optional[int] = None) -> str:
        """
        Build the context above the function to complete.
        :param metadata: dict, the metadata of target function.
        :param context_length: int or None, the number of lines of code context if provided.
        :return: str, a block of the context above.
        """
        file_path = os.path.join(self.repo_base_dir, metadata['relative_path'])
        function_name = metadata['namespace'].split('.')[-1]
        code = FileUtils.read_file_as_string(file_path)
        code_lines = code.splitlines()
        end_line_no = metadata['signature_position'][0]
        start_line_no = max(0, end_line_no - context_length) if context_length is not None else 0
        context_above_lines = code_lines[start_line_no:end_line_no]
        context_str = '\n'.join(context_above_lines)
        context_str = self.handle_extremely_long_field(context_str, Tokenizer(), max_length=4096, cut_from_tail=False)
        # return ''
        # return (f'\n# The contexts above the `{function_name}` function are:\n'
        #         f'```python\n{context_str}\n```\n\n')
        return (f'\n# The intra-file context above the `{function_name}` function is:\n'
                f'```python\n{context_str}\n```\n\n')

    def build_necessary_context_above(self, metadata) -> str:
        file_path = os.path.join(self.repo_base_dir, metadata['relative_path'])
        function_name = metadata['namespace'].split('.')[-1]
        code = FileUtils.read_file_as_string(file_path)
        code_lines = code.splitlines()
        end_line_no = metadata['signature_position'][0]
        context_above_lines = code_lines[end_line_no]
        context_str = '\n'.join(context_above_lines)
        context_str = self.handle_extremely_long_field(context_str, Tokenizer(), max_length=4096, cut_from_tail=False)
        return (f'\n# The contexts above the `{function_name}` function are:\n'
                f'```python\n{context_str}\n```\n\n')


    def build_completion(self, metadata):
        """
        Build the completion block using the signature and the requirements (Functionality and Arguments) of the target
        function.
        :param metadata: dict, the metadata of target function.
        :return: str, the string of completion block.
        """
        file_path = os.path.join(self.repo_base_dir, metadata['relative_path'])
        code = FileUtils.read_file_as_string(file_path)
        code_lines = code.splitlines()
        start_line_no, end_line_no = metadata['signature_position']
        signature_lines = code_lines[start_line_no:end_line_no]
        if self.bench == 'deveval':
            functionality = metadata['requirement']['Functionality']
            arguments = metadata['requirement']['Arguments']
            requirement = f"\"\"\"\n{functionality}\nInput-Output Arguments:\n{arguments}\n\"\"\""
        else:
            requirement = metadata['requirement']
        indent = metadata['indent']
        requirement_with_indent = textwrap.indent(requirement, ' ' * indent)
        return '```python\n' + '\n'.join(signature_lines) + '\n' + requirement_with_indent + '\n```\n'

    @staticmethod
    def handle_extremely_long_field(text, tokenizer, max_length=4096, cut_from_tail=True):
        """
        Truncate extremely long fields in the prompt.
        :param text: str, text to be truncate.
        :param tokenizer: Tokenizer, a specified tokenizer for the used model.
        :param max_length: int, the max length of the field.
        :param cut_from_tail: bool, cut from the tail of the input text if True.
        :return: str, truncated text if it is over-length.
        """
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_length:
            if cut_from_tail:
                return tokenizer.decode(tokens[:max_length])
            else:
                return tokenizer.decode(tokens[-max_length:])
        else:
            return text


class FewShotPromptBuilder(BasePromptBuilder):
    def __init__(self, num_of_demos, demo_path, bench='deveval'):
        super().__init__(bench=bench)
        self.separator = '# ' + '-' * 50
        self.num_of_demos = num_of_demos
        self.demo_path = demo_path

    def build_demonstrations(self):
        demo_list = []
        for i in range(self.num_of_demos):
            demo_path = self.demo_path.replace('.txt', f'_{i}.txt')
            demo = FileUtils.read_file_as_string(demo_path)
            demo_list.append(demo)
        return self.separator + '\n' + f'\n{self.separator}\n'.join(demo_list) + '\n' + self.separator + '\n'


class CoTPromptBuilder(FewShotPromptBuilder):
    """
    Prompt builder for query expansion. Also used as CoT/plan generator.
    """
    def __init__(self, metadata, num_of_demos, demo_path, include_code=False, bench='deveval'):
        super().__init__(num_of_demos, demo_path, bench)
        self.metadata = metadata
        self.include_code = include_code  # for CoT baseline

    def build_prompt_str(self):
        # add class information for methods
        if self.metadata['type'] != 'function':
            class_name = self.metadata['namespace'].split('.')[-2]
            function_type_desc = f'a method of `{class_name}` class, which is '
        else:
            function_type_desc = ''
        input_dict = {
            'function_name': self.metadata['namespace'].split('.')[-1],
            'demonstrations': self.build_demonstrations(),
            'context_above': self.build_context_above(self.metadata),
            'function_type_desc': function_type_desc,
            'function_signature': self.build_completion(self.metadata)
        }
        if self.include_code:
            return COT_AND_CODE_PROMPT_TEMPLATE['instruction_prompt'].format(**input_dict)
        else:
            return COT_PROMPT_TEMPLATE['instruction_prompt'].format(**input_dict)

    def build_prompt_line(self):
        if self.include_code:
            system_prompt = COT_AND_CODE_PROMPT_TEMPLATE['system_prompt']
        else:
            system_prompt = COT_PROMPT_TEMPLATE['system_prompt']
        return {
            'system_prompt': system_prompt,
            'instruction_prompt': self.build_prompt_str(),
            'metadata': self.metadata
        }


class DirectPromptBuilder(BasePromptBuilder):
    def __init__(self, metadata, context_length, infile_context=True, bench='deveval'):
        super().__init__(bench=bench)
        self.metadata = metadata
        self.context_length = context_length
        self.infile_context = infile_context

    def build_prompt_str(self):
        if self.infile_context:
            context_above = self.build_context_above(self.metadata, self.context_length)
        else:
            context_above = ''
        # add class information for methods
        if self.metadata['type'] != 'function':
            class_name = self.metadata['namespace'].split('.')[-2]
            function_type_desc = f'a method of `{class_name}` class, which is '
        else:
            function_type_desc = ''
        input_dict = {
            'function_name': self.metadata['namespace'].split('.')[-1],
            'file_path': self.metadata['relative_path'],
            'context_above': context_above,
            'function_type_desc': function_type_desc,
            'function_signature': self.build_completion(self.metadata)
        }
        return DIRECT_PROMPT_TEMPLATE['instruction_prompt'].format(**input_dict)


    def build_prompt_line(self, mode=None):
        prompt_line = {
            'system_prompt': DIRECT_PROMPT_TEMPLATE['system_prompt'],
            'instruction_prompt': self.build_prompt_str()
        }
        if mode == 'rag':
            prompt_line['query_window'] = dict()
            prompt_line['query_window']['metadata'] = self.metadata
        else:
            prompt_line['metadata'] = self.metadata
        return prompt_line


class LongContextPromptBuilder(BasePromptBuilder):
    def __init__(self, metadata, context_length, infile_context=True, bench='deveval'):
        super().__init__(bench=bench)
        self.metadata = metadata
        self.context_length = context_length
        self.infile_context = infile_context


    def build_repo_files(self):
        all_files = FileUtils.find_py_files(os.path.join(self.repo_base_dir, self.metadata['repo_path']))
        block_str = ''
        for f in all_files:
            if f == os.path.join(self.repo_base_dir, self.metadata['relative_path']):
                print("Skip target file.")
                continue
            code_lines = FileUtils.read_file_as_list(f)
            code_comment_lines = [f'# {line}' for line in code_lines]
            code_comment = f'# The code from `{f}` is:\n' + '\n'.join(code_comment_lines)
            block_str += code_comment + '\n' + '# ' + '-' * 50
        return block_str


    def build_prompt_str(self):
        repo_files = self.build_repo_files()
        if self.infile_context:
            context_above = self.build_context_above(self.metadata, self.context_length)
        else:
            context_above = ''
        # add class information for methods
        if self.metadata['type'] != 'function':
            class_name = self.metadata['namespace'].split('.')[-2]
            function_type_desc = f'a method of `{class_name}` class, which is '
        else:
            function_type_desc = ''
        input_dict = {
            'function_name': self.metadata['namespace'].split('.')[-1],
            'file_path': self.metadata['relative_path'],
            'repo_files': repo_files,
            'context_above': context_above,
            'function_type_desc': function_type_desc,
            'function_signature': self.build_completion(self.metadata)
        }
        return LONG_CONTEXT_PROMPT_TEMPLATE['instruction_prompt'].format(**input_dict)

    def build_prompt_line(self, mode=None):
        prompt_line = {
            'system_prompt': LONG_CONTEXT_PROMPT_TEMPLATE['system_prompt'],
            'instruction_prompt': self.build_prompt_str()
        }
        if mode == 'rag':
            prompt_line['query_window'] = dict()
            prompt_line['query_window']['metadata'] = self.metadata
        else:
            prompt_line['metadata'] = self.metadata
        return prompt_line


class SelfPlanningPromptBuilder(BasePromptBuilder):
    def __init__(self, metadata, signature, context_length, infile_context=True):
        super().__init__()
        self.metadata = metadata
        self.signature = signature
        self.context_length = context_length
        self.infile_context = infile_context

    def build_prompt_str(self):
        if self.infile_context:
            context_above = self.build_context_above(self.metadata, self.context_length)
        else:
            context_above = ''
        # add class information for methods
        if self.metadata['type'] != 'function':
            class_name = self.metadata['namespace'].split('.')[-2]
            function_type_desc = f'a method of `{class_name}` class, which is '
        else:
            function_type_desc = ''
        input_dict = {
            'function_name': self.metadata['namespace'].split('.')[-1],
            'context_above': context_above,
            'function_type_desc': function_type_desc,
            'function_signature': self.signature
        }
        return SELF_PLANNING_PROMPT_TEMPLATE['instruction_prompt'].format(**input_dict)

    def build_prompt_line(self):
        prompt_line = {
            'system_prompt': SELF_PLANNING_PROMPT_TEMPLATE['system_prompt'],
            'instruction_prompt': self.build_prompt_str(),
            'metadata': self.metadata
        }
        return prompt_line


class RAGPromptBuilder(BasePromptBuilder):
    def __init__(self, retrieved_result, bench='deveval'):
        super().__init__(bench)
        self.tokenizer = Tokenizer()
        self.retrieved_result = retrieved_result
        self.metadata = retrieved_result['metadata']
        self.separator = '# ' + '-' * 50
        # TODO: some hyperparameters need to extract
        # self.num_of_retrieved = 20
        # self.max_retrieval_length = 6000
        self.num_of_retrieved = 10
        self.max_retrieval_length = 4000
        self.context_length = None

    def build_single_block(self, item):
        metadata = item['metadata']
        # relative_path_from_repo = '/'.join(metadata['relative_path'].split('/')[2:])
        relative_path_from_repo = os.path.relpath(metadata['relative_path'], metadata['repo_path'])
        function_name = metadata['namespace'].split('.')[-1]
        path_comment = f'# The below function `{function_name}` can be found in:\n# {relative_path_from_repo}'
        code = DataUtils.get_code_by_metadata(metadata, self.repo_base_dir)
        # code = DataUtils.remove_multiline_comments(code)  # comment may also improve performance
        code_lines = code.splitlines()
        code_comment_lines = [f'# {line}' for line in code_lines]
        code_comment = f'# The code of `{function_name}` is:\n' + '\n'.join(code_comment_lines)
        block_str = path_comment + '\n' + self.separator + '\n' + code_comment + '\n' + self.separator
        tokenized_block = self.tokenizer.tokenize(block_str)
        block_len = len(tokenized_block)
        return block_str, block_len

    def build_retrieved_blocks(self):
        beginning_comment = ('# Here are some relevant code snippets from the repository to help you complete '
                             'the target function:\n')
        retrieval_length = len(self.tokenizer.tokenize(beginning_comment))
        retrieved_blocks = []
        chosen_windows = []
        for retrieved in self.retrieved_result['top_k_retrieved']:
            if len(chosen_windows) >= self.num_of_retrieved:
                break
            block_str, block_len = self.build_single_block(retrieved)
            if block_str is None:  # the context block is unavailable (for repocoder)
                continue
            if retrieval_length + block_len <= self.max_retrieval_length:
                chosen_windows.append(retrieved)
                ### change order ###
                # retrieved_blocks.insert(0, block_str)   # sort by similarity from lower to higher
                retrieved_blocks.append(block_str)  # reverse
                retrieval_length += block_len
        return (beginning_comment + '\n' + self.separator + '\n' + '\n'.join(retrieved_blocks),
                chosen_windows)

    def build_prompt_str(self):
        function_name = self.metadata['namespace'].split('.')[-1]
        retrieved_blocks, chosen_windows = self.build_retrieved_blocks()
        # context_above = self.build_context_above(self.metadata, self.context_length)
        context_above = ''
        if self.metadata['type'] == 'method':
            class_name = self.metadata['namespace'].split('.')[-2]
            function_type_desc = f'a method of `{class_name}` class, which is '
        else:
            function_type_desc = ''
        input_dict = {
            'function_name': function_name,
            'retrieved_blocks': retrieved_blocks,
            'context_above': context_above,
            'function_type_desc': function_type_desc,
            'function_signature': self.build_completion(self.metadata)
        }
        return RAG_PROMPT_TEMPLATE['instruction_prompt'].format(**input_dict), chosen_windows

    def build_prompt_line(self):
        prompt_line = dict()
        prompt_str, chosen_windows = self.build_prompt_str()
        prompt_line['system_prompt'] = RAG_PROMPT_TEMPLATE['system_prompt']
        prompt_line['instruction_prompt'] = prompt_str
        prompt_line['query_window'] = {
            'content': self.retrieved_result['content'],
            'metadata': self.retrieved_result['metadata']
        }
        return prompt_line


class RepoCoderPromptBuilder(RAGPromptBuilder):
    def __init__(self, retrieved_result, need_shift, bench='deveval'):
        super().__init__(retrieved_result, bench)
        self.need_shift = need_shift
        self.bench = bench


    def get_available_context_after_shift(self, query_metadata, retrieved_metadata):
        available_contexts = []
        query_relative_path = query_metadata['relative_path']
        for metadata in retrieved_metadata:
            file_path = os.path.join(self.repo_base_dir, metadata['relative_path'])
            try:
                retrieved_code = FileUtils.read_file_as_string(file_path)
            except FileNotFoundError as e:
                print(e)
                continue
            retrieved_code_lines = retrieved_code.splitlines()
            end_line_no = metadata['end_line_no']
            window_size = metadata['window_size']
            slice_size = metadata['slice_size']
            new_end_line_no = min(end_line_no + window_size // slice_size, len(retrieved_code_lines))
            new_start_line_no = max(0, new_end_line_no - window_size)
            content_lines = retrieved_code_lines[new_start_line_no:new_end_line_no]
            # filter unavailable context after shift
            if (metadata['relative_path'] == query_relative_path and
                    new_end_line_no >= query_metadata['signature_position'][0]):
                continue
            available_contexts.append(content_lines)
        return available_contexts

    def build_single_block(self, item):
        metadata = item['metadata']  # the metadata of repocoder is a list
        prefix_len = 2 if self.bench == 'deveval' else 1
        split_paths = [m['relative_path'].split(os.sep)[prefix_len:] for m in metadata]
        relative_paths_from_repo = [os.path.join(*split_path) for split_path in split_paths]
        path_comment = f'# The below code fragment can be found in:\n'
        path_comment += '\n'.join([f'# {relative_path}' for relative_path in relative_paths_from_repo])
        # shift down by a few lines to address the gap between retrieved context and the completion target
        if self.need_shift:
            available_contexts = self.get_available_context_after_shift(self.metadata, metadata)
            # if len(available_contexts) == 0:
            #     return None, None
            if len(available_contexts) < len(metadata):
                return None, None
            content_lines = available_contexts[0]  # choose the first available context
        else:
            content_lines = item['content'].splitlines()
        code_comment_lines = [f'# {content_line}' for content_line in content_lines]
        code_comment = '\n'.join(code_comment_lines)
        block_str = (path_comment + '\n' + self.separator + '\n' +
                     code_comment + '\n' + self.separator)
        tokenized_block = self.tokenizer.tokenize(block_str)
        block_len = len(tokenized_block)
        return block_str, block_len


class DracoPromptBuilder(BasePromptBuilder):
    def __init__(self, metadata, retrieved, bench='deveval'):
        super().__init__(bench=bench)
        self.metadata = metadata
        self.retrieved = retrieved
        self.context_length = None

    def build_prompt_str(self):
        function_name = self.metadata['namespace'].split('.')[-1]
        context_above = self.build_context_above(self.metadata, self.context_length)
        if self.metadata['type'] != 'function':
            class_name = self.metadata['namespace'].split('.')[-2]
            function_type_desc = f'a method of `{class_name}` class, which is '
        else:
            function_type_desc = ''
        input_dict = {
            'function_name': function_name,
            'retrieved_blocks': self.retrieved,
            'context_above': context_above,
            'function_type_desc': function_type_desc,
            'function_signature': self.build_completion(self.metadata)
        }
        return RAG_PROMPT_TEMPLATE['instruction_prompt'].format(**input_dict)

    def build_prompt_line(self):
        prompt_line = dict()
        prompt_str = self.build_prompt_str()
        prompt_line['system_prompt'] = RAG_PROMPT_TEMPLATE['system_prompt']
        prompt_line['instruction_prompt'] = prompt_str
        prompt_line['query_window'] = {
            'metadata': self.metadata
        }
        return prompt_line


if __name__ == '__main__':
    import tqdm
    bench = 'deveval'  # 'codereval'
    tasks = FileUtils.read_jsonl(Globals.OUR_BENCHMARK_METADATA_PATH)
    # tasks = FileUtils.read_jsonl(Globals.FULL_BENCHMARK_METADATA_PATH)
    # tasks = FileUtils.read_jsonl(Globals.CODEREVAL_BENCHMARK_METADATA_PATH)

    # Generate DevEval-cf for GraphCoder
    cf_namespaces = [d['namespace'] for d in tasks]
    prompts = FileUtils.read_jsonl(os.path.join(Globals.PROMPT_DIR, 'graphcoder', 'deveval.jsonl'))
    # prompts = FileUtils.read_jsonl(os.path.join(Globals.PROMPT_DIR, 'draco', 'draco-cf.jsonl'))
    prompt_lines = []
    for p in prompts:
        if p['metadata']['namespace'] in cf_namespaces:
        # if p['query_window']['metadata']['namespace'] in cf_namespaces:
            prompt_lines.append(p)
    prompt_path = os.path.join(Globals.PROMPT_DIR, 'graphcoder', 'deveval-cf.jsonl')
    FileUtils.dump_jsonl(prompt_lines, prompt_path)

    ## Generate long context benchmark for evaluation
    # prompt_lines = []
    # for task in tqdm.tqdm(tasks, total=len(tasks)):
    #     if task['repo_path'] not in ['Utilities/whereami', 'Utilities/pymusic-dl', 'Utilities/mackup', 'Database/bplustree', 'System/exodus-bundler', 'Security/trailscraper', 'Text-Processing/mistune', 'Security/threatingestor', 'System/flower', 'Database/litecli', 'Communications/hl7', 'Internet/sumy', 'Utilities/PyJWT', 'Communications/ehforwarderbot', 'Text-Processing/pymorphy2', 'Multimedia/hypertools', 'Communications/IMAPClient', 'Text-Processing/feedparser', 'Text-Processing/PyLaTeX', 'System/prometheus-client', 'System/viztracer', 'Communications/chatette', 'System/wal-e', 'Internet/python-twitter', 'Security/python-taint']:
    #         continue
    #     prompt_line = LongContextPromptBuilder(task, None, bench=bench).build_prompt_line()
    #     prompt_lines.append(prompt_line)
    # prompt_path = os.path.join(Globals.PROMPT_DIR, 'LCM', 'long-context.jsonl')
    # FileUtils.dump_jsonl(prompt_lines, prompt_path)
    #
    # proms = FileUtils.read_jsonl(os.path.join(Globals.PROMPT_DIR, 'logicoder', 'logicoder-reverse.jsonl'))
    # prompt_lines = []
    # for p in proms:
    #     if p['query_window']['metadata']['repo_path'] not in ['Utilities/whereami', 'Utilities/pymusic-dl', 'Utilities/mackup', 'Database/bplustree', 'System/exodus-bundler', 'Security/trailscraper', 'Text-Processing/mistune', 'Security/threatingestor', 'System/flower', 'Database/litecli', 'Communications/hl7', 'Internet/sumy', 'Utilities/PyJWT', 'Communications/ehforwarderbot', 'Text-Processing/pymorphy2', 'Multimedia/hypertools', 'Communications/IMAPClient', 'Text-Processing/feedparser', 'Text-Processing/PyLaTeX', 'System/prometheus-client', 'System/viztracer', 'Communications/chatette', 'System/wal-e', 'Internet/python-twitter', 'Security/python-taint']:
    #         continue
    #     prompt_lines.append(p)
    # prompt_path = os.path.join(Globals.PROMPT_DIR, 'logicoder', 'long-context.jsonl')
    # FileUtils.dump_jsonl(prompt_lines, prompt_path)



