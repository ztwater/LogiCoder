from collections import defaultdict
import json
import glob
import logging
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import textwrap
import tiktoken

from globals import Globals
# Add the source directory to sys.path
sys.path.append(os.path.join(Globals.ROOT_DIR, 'parser'))
from pyan_zyf_v2.analyzer import CallGraphVisitor


class Tokenizer:
    def __init__(self, tokenizer_name="p50k_base"):
        """
        Default tokenizer is p50k_base (for GPT-3 family). Other tokenizers include r50k_base (for GPT-2), p50k_edit
        (for text-davinci-edit-001) and cl100k_base (for GPT-4 and text-embedding-ada-002).
        """
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

    def tokenize(self, text):
        """
        Encode the token sequence without adding special tokens for text processing purpose. For model input, please use
         `encode` instead.
        """
        return self.tokenizer.encode_ordinary(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


class FileUtils:
    @staticmethod
    def read_json(file_path):
        with open(file_path, 'r') as f:
            content = json.load(f)
        return content

    @staticmethod
    def dump_json(obj, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(obj, f, indent=4)

    @staticmethod
    def read_jsonl(file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                js = json.loads(line)
                data.append(js)
        return data

    @staticmethod
    def dump_jsonl(obj, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf8') as f:
            for item in obj:
                f.write(json.dumps(item) + '\n')

    @staticmethod
    def read_file_as_string(file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            res = f.read()
        return res

    @staticmethod
    def read_file_as_list(file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        return [l.strip('\n') for l in lines]

    @staticmethod
    def write_file(content, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def load_pickle(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def dump_pickle(obj, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def get_repositories(bench='deveval'):
        """
        Return all the absolute paths of all repositories in DevEval.
        """
        if bench == 'deveval':
            repos = []
            for topic in os.listdir(Globals.REPO_BASE_DIR):
                subdir = os.path.join(Globals.REPO_BASE_DIR, topic)
                repos.extend([os.path.join(subdir, repo_name) for repo_name in os.listdir(subdir)])
            return repos
        else:
            repos = set()
            metadata = FileUtils.read_jsonl(Globals.CODEREVAL_BENCHMARK_METADATA_PATH)
            for d in metadata:
                repos.add(os.path.join(Globals.REPO_BASE_CODEREVAL_DIR, d['repo_path']))
            return list(repos)

    @staticmethod
    def iterate_repository(repo_path):
        pattern = os.path.join(f'{repo_path}', "**", "*.py")
        files = glob.glob(pattern, recursive=True)

        skipped_files = []
        loaded_code_files = dict()
        for file_name in files:
            try:
                code = FileUtils.read_file_as_string(file_name)
                relative_path = PathUtils.get_relative_path(file_name)
                loaded_code_files[relative_path] = code
            except Exception as e:
                skipped_files.append((file_name, e))
                continue

        if len(skipped_files) > 0:
            print(f"Skipped {len(skipped_files)} out of {len(files)} files due to I/O errors")
            for file_name, e in skipped_files:
                print(f"{file_name}: {e}")
        return loaded_code_files

    @staticmethod
    def find_py_files(folder):
        py_files = []
        # Directories to exclude (non-source code)
        exclude_dirs = {
            # Virtual environments
            'venv', 'env', '.venv', '.env', 'myenv',
            # Python cache and compiled
            '__pycache__', '.pytest_cache', '.mypy_cache', '.hypothesis',
            # Build and distribution
            'build', 'dist', 'htmlcov',
            # Package metadata
            '.egg-info', '.eggs',
            # Testing and CI
            '.tox', '.coverage',
            # Version control
            '.git', '.svn', '.hg',
            # Node (if any JS files)
            'node_modules',
            # Package installations
            'site-packages', 'dist-packages',
            # IDE and editor
            '.vscode', '.idea', '.sublime-project',
        }

        for root, dirs, files in os.walk(folder):
            # Filter out excluded directories in-place to prevent os.walk from descending
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]

            for file in files:
                if file.endswith(".py"):
                    # Remove intermediate cloned files from CoderEval
                    if 'passk_validte' in file:
                        continue
                    py_files.append(os.path.join(root, file))


        return py_files

    @staticmethod
    def clean_tmp_files(target_dir):
        for file in os.listdir(target_dir):
            file_path = os.path.join(target_dir, file)
            if re.search(r'batch\d+\.jsonl$', file) and os.path.isfile(file_path):
                os.remove(file_path)


class PathUtils:
    @staticmethod
    def get_relative_path(file_path):
        base_dir_idx = len(os.path.normpath(Globals.REPO_BASE_DIR).split(os.sep))
        return os.path.join(*os.path.normpath(file_path).split(os.sep)[base_dir_idx:])

    @staticmethod
    def get_repo_window_path(repo, window_size, slice_size):
        repo_name = os.path.basename(repo)
        output_path = os.path.join(Globals.REPO_WINDOW_DIR, f'{repo_name}_ws{window_size}_ss{slice_size}.pkl')
        return output_path

    @staticmethod
    def get_benchmark_window_path(repo, window_size):
        repo_name = os.path.basename(repo)
        output_path = os.path.join(Globals.BENCHMARK_WINDOW_DIR, f'{repo_name}_ws{window_size}.pkl')
        return output_path

    @staticmethod
    def get_ground_truth_window_path(repo, window_size):
        repo_name = os.path.basename(repo)
        output_path = os.path.join(Globals.GROUND_TRUTH_WINDOW_DIR, f'{repo_name}_ws{window_size}.pkl')
        return output_path

    @staticmethod
    def get_prediction_window_path(repo, prediction_path):
        repo_name = os.path.basename(repo)
        prediction_name = os.path.basename(prediction_path)
        output_path = os.path.join(Globals.PREDICTION_WINDOW_DIR, f'{repo_name}_{prediction_name[:-6]}.pkl')
        return output_path

    @staticmethod
    def get_req_window_path(repo):
        repo_name = os.path.basename(repo)
        output_path = os.path.join(Globals.REQ_WINDOW_DIR, f'{repo_name}_req.pkl')
        return output_path

    @staticmethod
    def get_requirement_window_path(repo):
        repo_name = os.path.basename(repo)
        output_path = os.path.join(Globals.REQ_WINDOW_DIR, f'{repo_name}.pkl')
        return output_path

    @staticmethod
    def get_vector_path(window_path, vectorizer_name):
        vector_path = window_path.replace('/window/', '/vector/')
        vector_path = vector_path.replace('.pkl', f'_{vectorizer_name}.pkl')
        vector_path = vector_path.replace('.jsonl', f'_{vectorizer_name}.pkl')
        return vector_path

    @staticmethod
    def get_retrieval_path(query_vector_path, top_k):
        file_name = os.path.basename(query_vector_path)[:-4]
        output_path = os.path.join(Globals.RETRIEVAL_DIR, f'{file_name}_top{top_k}.pkl')
        return output_path

    # ============================== Completion Path ==============================
    @staticmethod
    def get_completion_path(args):
        sub_dir, mode, prompt_name, model_str, T, N = args
        if sub_dir:
            return os.path.join(Globals.PREDICTION_DIR, sub_dir,
                                f'completion_{mode}_{prompt_name}_{model_str}_T{T}_N{N}.jsonl')
        return os.path.join(Globals.PREDICTION_DIR, f'completion_{mode}_{prompt_name}_{model_str}_T{T}_N{N}.jsonl')

class DataUtils:
    @staticmethod
    def remove_inline_comments(line):
        result = []
        in_single_quote = False
        in_double_quote = False

        for char_index, char in enumerate(line):
            if char == "#" and not in_single_quote and not in_double_quote:
                break  # Stop when a '#' is found outside of quotes
            elif char == "'":
                in_single_quote = not in_single_quote
            elif char == '"':
                in_double_quote = not in_double_quote
            result.append(char)

        return "".join(result)

    @staticmethod
    def remove_all_comments(code):
        multiline_comment_pattern = r'(\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\")'
        # remove multiline comments
        method_code = re.sub(multiline_comment_pattern, '', code, flags=re.DOTALL)
        # split the code into lines
        lines = method_code.split('\n')
        new_lines = []
        for line in lines:
            new_lines.append(DataUtils.remove_inline_comments(line))
        # remove empty lines
        new_lines = [line for line in new_lines if line.strip() != '']
        return '\n'.join(new_lines)

    @staticmethod
    def remove_multiline_comments(code):
        multiline_comment_pattern = r'(\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\")'
        method_code = re.sub(multiline_comment_pattern, '', code, flags=re.DOTALL)
        return method_code

    @staticmethod
    def get_leading_spaces(string):
        return len(string) - len(string.lstrip())

    @staticmethod
    def extract_function_signature(code_str):
        """
        Extract the signature from a function.
        :param code_str: str, the code string.
        :return: str, the function signature string.
        """
        pattern = r"def\s+\w+\s*\([^)]*\)[^:]*:"
        signature = re.search(pattern, code_str, re.MULTILINE)
        end_pos = signature.end() - 1
        return code_str[:end_pos]

    @staticmethod
    def extract_class_signature(code_str):
        """
        Extract the signature from a class.
        :param code_str: str, the code string.
        :return: str, the class signature string.
        """
        pattern = r"class\s+[^:]*:"
        signature = re.search(pattern, code_str, re.MULTILINE)
        end_pos = signature.end() - 1
        return code_str[:end_pos]

    @staticmethod
    def extract_function_from_output_old(output, func_name):
        code_list = output.splitlines()
        func_code_list = []
        is_this_method = False
        leading_space = 0
        func_def_prefix = 'def ' + func_name + '('
        for i, line in enumerate(code_list):
            if func_def_prefix in line:
                is_this_method = True
                leading_space = DataUtils.get_leading_spaces(line)
                func_code_list.append(line[leading_space:])
            elif is_this_method:
                if (DataUtils.get_leading_spaces(line) > leading_space
                        or len(line) == 0 or all(char.isspace() for char in line)):
                    func_code_list.append(line[leading_space:])
                else:
                    break
        func_code = '\n'.join(func_code_list)
        return func_code

    @staticmethod
    def extract_function_from_output(output, func_name):
        if output is None:
            return ''
        code_list = output.splitlines()
        func_code_list = []
        is_this_method = False
        is_sig = False
        leading_space = 0
        func_def_prefix = 'def ' + func_name + '('
        for i, line in enumerate(code_list):
            if func_def_prefix in line:
                if ')' in func_def_prefix:
                    is_this_method = True
                else:
                    is_sig = True
                leading_space = DataUtils.get_leading_spaces(line)
                func_code_list.append(line[leading_space:])
            elif is_this_method:
                if (DataUtils.get_leading_spaces(line) > leading_space
                        or len(line) == 0 or all(char.isspace() for char in line)):
                    func_code_list.append(line[leading_space:])
                else:
                    break
            elif is_sig:
                func_code_list.append(line[leading_space:])
                if ')' in line:
                    is_sig = False
                    is_this_method = True
        func_code = '\n'.join(func_code_list)
        return func_code

    @staticmethod
    def extract_completion(code, indent=4):
        signature_pattern = r'^def .*:\n'
        multiline_pattern = r'(\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\")'
        code = re.sub(signature_pattern, '', code, flags=re.MULTILINE)
        code = re.sub(multiline_pattern, '', code, flags=re.DOTALL)
        code = re.sub(r'```python\n(.*?)```', r'\1', code, flags=re.DOTALL)
        dedented_code = textwrap.dedent(code)
        indented_code = textwrap.indent(dedented_code, ' ' * indent)
        return indented_code

    @staticmethod
    def extract_cot(output):
        pattern = r'(?s)(?:\d+\.(.*))+'
        # search for the first match in the LLM output
        match = re.search(pattern, output, re.MULTILINE)
        if match:
            matched_text = match.group(0)
            # define the regex pattern to match individual numbered steps
            step_pattern = r'\d+\.(.*)'
            steps = re.findall(step_pattern, matched_text)
            steps = [s.strip() for s in steps]
            return '\n'.join(steps)
        else:
            print("No CoT found.")

    @staticmethod
    def update_metadata(metadata):
        """
        Update some keys and values of DevEval task metadata to improve its consistency.
        :param metadata: dict, the DevEval task metadata.
        :return: dict, an improved metadata format used in our approach.
        """
        return {
            'namespace': metadata['namespace'],
            'type': metadata['type'],
            'repo_path': metadata['project_path'],
            'relative_path': metadata['completion_path'],
            'signature_position': [metadata['signature_position'][0] - 1, metadata['signature_position'][1]],
            'body_position': [metadata['body_position'][0] - 1, metadata['body_position'][1]],
            'dependency': metadata['dependency'],
            'requirement': metadata['requirement'],
            'tests': metadata['tests'],
            'indent': metadata['indent']
        }

    @staticmethod
    def get_code_by_metadata(metadata, repo_base_dir):
        """
        Get the code snippet by its start and end line numbers by metadata.
        :param metadata: dict, the metadata of knowledge.
        :return: str, the code string.
        """
        file_path = os.path.join(repo_base_dir, metadata['relative_path'])
        code = FileUtils.read_file_as_string(file_path)
        code_lines = code.splitlines()
        start_line_no, end_line_no = metadata['start_line_no'], metadata['end_line_no']
        selected_lines = code_lines[start_line_no:end_line_no]
        return '\n'.join(selected_lines)

    @staticmethod
    def get_project_structure(target_item, relevant_items):
        """
        Generate a simplified project structure using namespaces.
        :param target_item: dict, the target function to be generated (with its metadata).
        :param relevant_items: list, a list of items to annotate their relative path to the target function.
        :return: str, an ASCII-style tree string show the project structure.
        """
        def tree():
            return defaultdict(tree)

        def get_paths(namespace, file_path):
            file_name = os.path.basename(file_path)
            # add __init__.py file to the namespace (dropped by DevEval)
            if file_name == '__init__.py':
                module_rel_path = os.path.relpath(os.path.dirname(file_path), Globals.REPO_BASE_DIR)
                # remove topic/repo path
                module_name = os.path.basename(module_rel_path)
                module_prefix = '.'.join(module_rel_path.split(os.sep)[2:])
                try:
                    _, s = namespace.split(module_prefix)
                except Exception as e:
                    # DevEval contains some wrong namespaces (missing "docs" or "tests" directories)
                    # print(f'Exception found when adding __init__ file to {namespace}, '
                    #       f'which does not match the prefix {module_prefix}.')
                    # print(e)
                    s = '.' + namespace.split(module_name + '.')[-1]
                namespace = module_prefix + '.__init__' + s
            parts = namespace.split('.')
            return file_name, parts

        def add_path(root, file_name, parts):
            in_file = False
            for p in parts:
                # dfs, iteratively build path
                if not in_file:
                    if p == file_name[:-3]:
                        root = root[f'{p}.py']  # add .py file
                        in_file = True
                    else:
                        root = root[f'{p}/']  # add directory
                else:
                    root = root[p]

        # root node
        root = tree()

        # iteratively add all relevant items
        for item in relevant_items:
            namespace = item['metadata']['namespace']
            file_path = os.path.join(Globals.REPO_BASE_DIR, item['metadata']['relative_path'])
            file_name, parts = get_paths(namespace, file_path)
            parts[-1] = parts[-1] + f" (type: {item['metadata']['type']})"
            add_path(root, file_name, parts)

        namespace = target_item['metadata']['namespace']
        file_path = os.path.join(Globals.REPO_BASE_DIR, target_item['metadata']['relative_path'])
        file_name, parts = get_paths(namespace, file_path)
        target_str = '[TARGET_FUNCTION] ' if AnalUtils.is_function(target_item['metadata']['type']) \
            else '[TARGET_CLASS] '
        parts[-1] = target_str + parts[-1]
        add_path(root, file_name, parts)

        def tree_to_string(tree, indent=0):
            s = ""
            for key, value in sorted(tree.items()):
                if indent > 0:
                    s += '|   ' * (indent - 1) + '|---'
                s += key + '\n'
                if isinstance(value, dict):
                    s += tree_to_string(value, indent + 1)
            return s

        return tree_to_string(root)

    @staticmethod
    def encapsulate_metadata_in_item(metadata):
        return {'metadata': metadata}

class AnalUtils:
    @staticmethod
    def get_call_graph_visitor(repo_path):
        # get all python files
        py_files = FileUtils.find_py_files(repo_path)
        filenames = []
        for fn in py_files:
            for fn2 in glob.glob(fn, recursive=True):
                abs_fn2 = os.path.abspath(fn2)
                filenames.append(abs_fn2)
        # build call graph and return the visitor
        return CallGraphVisitor(filenames, root=None)


    @staticmethod
    def is_concrete_node(node):
        if node.defined and node.namespace is not None and \
                node.ast_node is not None and hasattr(node.ast_node, 'lineno'):
            return True
        return False

    @staticmethod
    def get_deps(visitor, caller_node):
        """
        Get the dependencies for a single node.
        :param visitor: CallGraphVisitor, the call graph visitor parsed by pyan.
        :param caller_node: Node, a function or class to be analyzed.
        :return: dict, dependencies of the provided node.
        """
        deps = {
            'intra_class': [],
            'intra_file': [],
            'cross_file': []
        }
        for node_name in visitor.uses_edges:
            if node_name != caller_node.get_name():
                continue
            caller_file = caller_node.filename
            caller_class = AnalUtils.get_class(caller_node)
            for callee_node in visitor.uses_edges[node_name]:
                # (?) callee node is also in the current repository
                if callee_node.namespace is not None and callee_node.namespace != "*":
                    callee_file = callee_node.filename
                    callee_class = AnalUtils.get_class(callee_node)
                    # print('Caller:', node_name, caller_file, caller_class)
                    # print('Callee:', callee_node.get_name(), callee_file, callee_class)
                    # if caller is the parent of the callee, it just calls its own member
                    if callee_file == caller_file and node_name in callee_node.namespace:
                        pass
                    elif callee_class == caller_class and callee_class is not None:
                        deps['intra_class'].append(callee_node)
                    elif callee_file == caller_file:
                        deps['intra_file'].append(callee_node)
                    else:
                        deps['cross_file'].append(callee_node)
        return deps

    @staticmethod
    def get_node_info(node, bench='deveval'):
        if bench == 'deveval':
            repo_base_dir = Globals.REPO_BASE_DIR
            repo_path_len = 2
        else:
            repo_base_dir = Globals.REPO_BASE_CODEREVAL_DIR
            repo_path_len = 1
        file_name = node.filename
        node_class = AnalUtils.get_class(node)
        relative_path = os.path.relpath(file_name, repo_base_dir)
        repo_path = os.path.join(*relative_path.split(os.sep)[:repo_path_len])
        start_line_no, end_line_no = AnalUtils.get_position(node)
        return {
            'namespace': node.get_name(),
            'type': node.flavor.value,
            'class': node_class,
            'repo_path': repo_path,
            'relative_path': relative_path,
            'start_line_no': start_line_no,
            'end_line_no': end_line_no
        }

    @staticmethod
    def get_node_metadata(node):
        file_name = node.filename
        relative_path = os.path.relpath(file_name, Globals.REPO_BASE_DIR)
        repo_path = os.path.join(*relative_path.split(os.sep)[:2])
        code_str = FileUtils.read_file_as_string(file_name)
        code_lines = code_str.splitlines()
        start_line_no, end_line_no = AnalUtils.get_position(node)
        return {
            'namespace': node.get_name(),
            'type': node.flavor.value,
            'repo_path': repo_path,
            'relative_path': relative_path,
            'start_line_no': start_line_no,
            'end_line_no': end_line_no,
            'code': '\n'.join(code_lines[start_line_no:end_line_no])
        }

    @staticmethod
    def get_position(node):
        if node.ast_node is not None and hasattr(node.ast_node, 'lineno'):
            return [node.ast_node.lineno - 1, node.ast_node.end_lineno]
        # return 0, 0 for other nodes
        print(f'No line number found for {node.get_name()}.')
        return [0, 0]

    @staticmethod
    def is_function(node_type):
        if node_type in ['method', 'staticmethod', 'classmethod', 'propertymethod', 'function']:
            return True
        return False

    # @staticmethod
    # def get_class(node):
    #     node_type = node.flavor.value
    #     if node_type in ['method', 'staticmethod', 'classmethod', 'propertymethod']:
    #         return node.namespace.split(".")[-1]  # get the class of the method
    #     elif node_type == "function":
    #         return None
    #     elif node_type == "class":
    #         return node.name
    #     elif node_type == "module":
    #         return None
    #     else:
    #         return node.namespace.split(".")[-1]  # default return or other objects

    @staticmethod
    def get_class(node):
        node_type = node.flavor.value
        if node_type in ['method', 'staticmethod', 'classmethod', 'propertymethod']:
            return node.namespace  # get the class of the method
        elif node_type == "function":
            return None
        elif node_type == "class":
            return node.get_name()
        elif node_type == "module":
            return None
        else:
            return node.namespace  # default return or other objects

def run_process(cmd, cwd, shell=False):
    if shell:
        print(f"Running {cmd}.")
    else:
        print(f"Running {' '.join(cmd)}.")
    process = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    output, errors = process.communicate()
    if process.returncode != 0:
        print("Error:", errors.decode())
        # raise Exception
    else:
        pass
        # print("Output:", output.decode())
    return output


def silent_non_app_loggers(prefixes=('ztwater')):
    for logger_name in logging.Logger.manager.loggerDict:
        if not logger_name:  # skip root
            continue

        is_app_logger = any(
            logger_name.startswith(prefix) for prefix in prefixes
        )

        if not is_app_logger:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            # Optional: prevent propagation to your handlers
            logger.propagate = False

def parse_code_element(data):
    relpath = os.path.relpath(Path(data['relative_path']), Path(data['repo_path']))
    parts = data['namespace'].split('.')
    func_name = parts[-1]
    if data['type'] == 'method':
        class_name = parts[-2]
        ele_id = f"{class_name}.{func_name}"
    else:
        class_name = None
        ele_id = func_name
    return {
        'id': f"{relpath}::{ele_id}",
        'file_path': relpath,
        'class_name': class_name,
        'func_name': func_name
    }