import argparse
import itertools
import os
import time
import tqdm

from build_prompt import DirectPromptBuilder, RAGPromptBuilder
from build_vector import BagOfWordVectorizer, UniXCoderVectorizer
from build_window import BenchmarkWindowBuilder, RequirementWindowBuilder
from globals import Globals
from myutils import DataUtils, FileUtils, PathUtils, Tokenizer
from retriever import RepoCoderDenseRetriever, RepoCoderSparseRetriever

class Runner:
    def __init__(self, args, repos, benchmark_metadata):
        self.args = args
        self.repos = repos
        self.benchmark_metadata = benchmark_metadata

    def run_function_vectorization(self, bench='deveval'):
        repo_base_dir = Globals.REPO_BASE_DIR if bench == 'deveval' else Globals.REPO_BASE_CODEREVAL_DIR
        time_dict = {}
        for repo in self.repos:
            start_time = time.time()
            repo_name = os.path.basename(repo)
            func_path = os.path.join(Globals.CANDIDATE_DIR, f'{repo_name}_funcs.json')
            ####
            vector_path = os.path.join(Globals.KNOWLEDGE_BASE_DIR, 'vector', 'func',
                                       f'{repo_name}_funcs_{self.args.vectorizer.get_name()}.pkl')
            if os.path.exists(vector_path):
                print(f'Vectors for {repo_name} have already been built.')
                continue
            ####
            funcs = FileUtils.read_json(func_path)
            if self.args.vectorizer.get_name() == 'bow':
                vectorizer = self.args.vectorizer(func_path, Tokenizer())
            elif self.args.vectorizer.get_name() == 'unixcoder':
                vectorizer = self.args.vectorizer(func_path, args.load_from_local)
            items = []
            for func_ns in tqdm.tqdm(funcs, total=len(funcs)):
                metadata = funcs[func_ns]
                file_path = os.path.join(repo_base_dir, metadata['relative_path'])
                code = FileUtils.read_file_as_string(file_path)
                code_lines = code.splitlines()
                start_line_no = metadata['start_line_no']
                end_line_no = metadata['end_line_no']
                func_code = '\n'.join(code_lines[start_line_no:end_line_no])
                embedding = vectorizer.vectorize(func_code)
                item = {
                    'content': func_code,
                    'metadata': metadata,
                    'data': [{'embedding': embedding}]
                }
                items.append(item)

            ####
            time_dict[repo_name] = time.time() - start_time
            FileUtils.dump_pickle(items, vector_path)
        FileUtils.dump_json(time_dict, os.path.join(Globals.ROOT_DIR, 'logs', 'latency', 'func_vec.json'))

    def run_function_context_vectorization(self, window_size, bench='deveval'):
        repo_base_dir = Globals.REPO_BASE_DIR if bench == 'deveval' else Globals.REPO_BASE_CODEREVAL_DIR
        for repo in self.repos:
            repo_name = os.path.basename(repo)
            func_path = os.path.join(Globals.CANDIDATE_DIR, f'{repo_name}_funcs.json')
            if not os.path.exists(func_path):
                continue
            funcs = FileUtils.read_json(func_path)
            if self.args.vectorizer.get_name() == 'bow':
                vectorizer = self.args.vectorizer(func_path, Tokenizer())
            elif self.args.vectorizer.get_name() == 'unixcoder':
                vectorizer = self.args.vectorizer(func_path, args.load_from_local)
            items = []
            for func_ns in tqdm.tqdm(funcs, total=len(funcs)):
                metadata = funcs[func_ns]
                file_path = os.path.join(repo_base_dir, metadata['relative_path'])
                code = FileUtils.read_file_as_string(file_path)
                code_lines = code.splitlines()
                ctx_end_line_no = metadata['start_line_no']
                ctx_start_line_no = max(0, ctx_end_line_no - window_size)
                context = '\n'.join(code_lines[ctx_start_line_no:ctx_end_line_no])
                embedding = vectorizer.vectorize(context)
                item = {
                    'content': context,
                    'metadata': metadata,
                    'data': [{'embedding': embedding}]
                }
                items.append(item)
            vector_path = os.path.join(Globals.KNOWLEDGE_BASE_DIR, 'vector', 'func',
                                       f'{repo_name}_funcs-ctx_{self.args.vectorizer.get_name()}.pkl')
            FileUtils.dump_pickle(items, vector_path)

    def run_benchmark_vectorization(self, window_size=50, bench='deveval'):
        for repo in self.repos:  # store tasks by repo
            BenchmarkWindowBuilder(self.benchmark_metadata, repo, window_size).build_windows(bench=bench)
            benchmark_window_path = PathUtils.get_benchmark_window_path(repo, window_size)
            if self.args.vectorizer.get_name() == 'bow':
                self.args.vectorizer(benchmark_window_path, Tokenizer()).build()
            elif self.args.vectorizer.get_name() == 'unixcoder':
                self.args.vectorizer(benchmark_window_path, args.load_from_local).build()

    def run_requirement_vectorization(self, bench='deveval'):
        for repo in self.repos:  # store tasks by repo
            RequirementWindowBuilder(self.benchmark_metadata, repo, 50).build_windows(bench=bench)
            requirement_window_path = PathUtils.get_requirement_window_path(repo)
            if self.args.vectorizer.get_name() == 'bow':
                self.args.vectorizer(requirement_window_path, Tokenizer()).build()
            elif self.args.vectorizer.get_name() == 'unixcoder':
                self.args.vectorizer(requirement_window_path, args.load_from_local).build()

    def run_retrieval(self):
        time_dict = {}
        for repo in self.repos:
            start_time = time.time()
            repo_name = os.path.basename(repo)
            # query_window_path = PathUtils.get_requirement_window_path(repo)
            query_window_path = PathUtils.get_benchmark_window_path(repo, 50)
            query_vector_path = PathUtils.get_vector_path(query_window_path, self.args.vectorizer.get_name())
            repo_vector_path = os.path.join(Globals.KNOWLEDGE_BASE_DIR, 'vector', 'func',
                                            f'{repo_name}_funcs_{self.args.vectorizer.get_name()}.pkl')
            # if not os.path.exists(repo_vector_path):
            #     continue
            query_lines = FileUtils.load_pickle(query_vector_path)
            repo_lines = FileUtils.load_pickle(repo_vector_path)
            retrieval_path = os.path.join(Globals.RETRIEVAL_DIR,
                                          f'{repo_name}_{self.args.vectorizer.get_name()}.pkl')
            if not os.path.exists(retrieval_path):
                if self.args.vectorizer.get_name() == 'bow':
                    retriever = RepoCoderSparseRetriever(repo, query_lines, repo_lines, self.args.top_k, retrieval_path)
                elif self.args.vectorizer.get_name() == 'unixcoder':
                    retriever = RepoCoderDenseRetriever(repo, query_lines, repo_lines, self.args.top_k, retrieval_path)
                retriever.retrieve()
            ####
            repo_name = os.path.basename(repo)
            time_dict[repo_name] = time.time() - start_time
        # FileUtils.dump_json(time_dict, os.path.join(Globals.ROOT_DIR, 'logs', 'latency', 'retrieve.json'))

    def build_prompts_for_knowledge_aware_rag(self, bench='deveval'):
        prompt_lines = []
        ns = [m['namespace'] for m in self.benchmark_metadata]
        for repo in self.repos:
            repo_name = os.path.basename(repo)
            retrieval_path = os.path.join(Globals.RETRIEVAL_DIR,
                                          f'{repo_name}_{self.args.vectorizer.get_name()}.pkl')
            # if not os.path.exists(retrieval_path):
            #     continue
            retrieved_results = FileUtils.load_pickle(retrieval_path)
            for retrieved_result in retrieved_results:
                if retrieved_result['metadata']['namespace'] not in ns:
                    continue
                prompt_line = RAGPromptBuilder(retrieved_result, bench=bench).build_prompt_line()
                prompt_lines.append(prompt_line)
            print(f"Build {len(prompt_lines)} knowledge-aware prompts for {len(retrieved_results)} tasks "
                  f"in {repo_name}.")
        # finished = [l['query_window']['metadata']['namespace'] for l in prompt_lines]
        # print(len(finished))
        # # add direct generation prompts
        # for metadata in self.benchmark_metadata:
        #     repo_path = os.path.join(Globals.REPO_BASE_DIR, metadata['repo_path'])
        #     if repo_path in self.repos and metadata['namespace'] not in finished:
        #         prompt_line = DirectPromptBuilder(metadata, None, True).build_prompt_line(mode='rag')
        #         prompt_lines.append(prompt_line)
        print(f"Build {len(prompt_lines)} prompts in total.")
        prompt_path = os.path.join(Globals.KA_PROMPT_DIR, f'knowledge_aware_{self.args.vectorizer.get_name()}-full.jsonl')
        FileUtils.dump_jsonl(prompt_lines, prompt_path)

    def get_ground_truth_deps(self):
        ground_truths = []
        for task in tqdm.tqdm(self.benchmark_metadata, total=len(self.benchmark_metadata)):
            line = {
                'content': '',
                'metadata': task,
                'top_k_retrieved': []
            }
            repo_name = os.path.basename(task['repo_path'])
            deps = itertools.chain.from_iterable(task['dependency'].values())
            for d in deps:
                func_path = os.path.join(Globals.CANDIDATE_DIR, f'{repo_name}_funcs.json')
                funcs = FileUtils.read_json(func_path)
                if d in funcs:
                    line['top_k_retrieved'].append(DataUtils.encapsulate_metadata_in_item(funcs[d]))
            if len(line['top_k_retrieved']) > 0:
                ground_truths.append(line)
        return ground_truths

    def build_prompts_for_gt_callee(self, augmented_content):
        prompt_lines = []
        for item in augmented_content:
            prompt_line = RAGPromptBuilder(item).build_prompt_line()
            prompt_lines.append(prompt_line)
        print(f"Build {len(prompt_lines)} prompts from rag pipeline.")
        # add direct generation prompts
        finished = [l['query_window']['metadata']['namespace'] for l in prompt_lines]
        print(finished)
        for metadata in self.benchmark_metadata:
            repo_path = os.path.join(Globals.REPO_BASE_DIR, metadata['repo_path'])
            if repo_path in self.repos and metadata['namespace'] not in finished:
                prompt_line = DirectPromptBuilder(metadata, None, True).build_prompt_line(mode='rag')
                prompt_lines.append(prompt_line)
        print(f"Build {len(prompt_lines)} prompts in total.")
        prompt_path = os.path.join(Globals.LOGICODER_PROMPT_DIR, 'logicoder_gt_callee.jsonl')
        FileUtils.dump_jsonl(prompt_lines, prompt_path)


def function_retrieval_workflow(args, repos, metadata, bench='deveval'):
    start_time = time.time()
    runner = Runner(args, repos, metadata)
    # Step 1: vectorize requirements
    # runner.run_benchmark_vectorization(window_size=50, bench=bench)
    # runner.run_requirement_vectorization(bench=bench)
    # Step 2: vectorize funcs
    # runner.run_function_vectorization(bench=bench)
    # runner.run_function_context_vectorization(window_size=20, bench=bench)
    vector_time = time.time() - start_time
    print(f'[TIMING] Vectorize all repos in {vector_time:.2f}s')
    # Step 3: retrieve
    # runner.run_retrieval()
    # Step 4: build prompt for generation
    runner.build_prompts_for_knowledge_aware_rag(bench=bench)

def logicoder_gt_callee_workflow(args, repos, metadata):
    runner = Runner(args, repos, metadata)
    ground_truth_deps = runner.get_ground_truth_deps()
    runner.build_prompts_for_gt_callee(ground_truth_deps)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectorizer_name', type=str, required=True)
    parser.add_argument('--load_from_local', action='store_true')
    parser.add_argument('--top_k', type=int, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # specify the vectorizer to use by their alias in arguments
    if args.vectorizer_name == 'bow':
        args.vectorizer = BagOfWordVectorizer
    elif args.vectorizer_name == 'unixcoder':
        args.vectorizer = UniXCoderVectorizer
    else:
        raise Exception('Unavailable vectorizer.')

    bench = 'deveval'  # 'codereval'  # 'deveval'

    # get current available repos
    repos = FileUtils.get_repositories(bench)
    print(f"Target repos ({len(repos)} in total):\n{repos}")

    if bench == 'deveval':
        # updated_metadata = FileUtils.read_jsonl(Globals.OUR_BENCHMARK_METADATA_PATH)
        updated_metadata = FileUtils.read_jsonl(Globals.FULL_BENCHMARK_METADATA_PATH)
    else:
        updated_metadata = FileUtils.read_jsonl(Globals.CODEREVAL_BENCHMARK_METADATA_PATH)

    # baselines
    function_retrieval_workflow(args, repos, updated_metadata, bench=bench)
    # logicoder_gt_callee_workflow(args, repos, updated_metadata)
