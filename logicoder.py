import argparse
import copy
import itertools
import os
import random
import time
import tqdm

from build_call_graph import CallGraphBuilder
from build_prompt import RAGPromptBuilder, DirectPromptBuilder
from extract_function import FunctionExtractor
from globals import Globals
from myutils import FileUtils, DataUtils, Tokenizer
from semantic_search_fusion import rerank, main
from repository_cache import RepositoryDataCache


def remove_percentage(s, p):
    """
    Simulate incorrect static analysis results.
    """
    n = len(s)
    k = int(n * p / 100)
    to_remove = set(random.sample(list(s), k))
    return s - to_remove


class UsageExampleRetriever:
    def __init__(self, repo_name, tasks, tokenizer=None, bench='deveval'):
        self.repo_name = repo_name
        self.tasks = tasks
        self.bench = bench
        self.repo_base_dir = Globals.REPO_BASE_DIR if self.bench == 'deveval' else Globals.REPO_BASE_CODEREVAL_DIR

        # Load repository data once using cache
        cache = RepositoryDataCache.get_instance()
        self.repo_data = cache.get_repository_data(self.repo_name, load_visitor=True, load_funcs=True,
                                                   load_called_graph=True)
        self.funcs = self.repo_data['funcs']
        self.called_graph = self.repo_data['called_graph']
        self.mappings = self.repo_data['mappings']

        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer('cl100k_base')

        # Cache for file contents to avoid redundant I/O
        self._file_cache = {}  # relative_path -> (code_str, code_lines)
        self._task_vector_cache = {}  # task_namespace -> task_vector

    def _get_file_content(self, relative_path):
        """Get file content with caching."""
        if relative_path not in self._file_cache:
            file_path = os.path.join(self.repo_base_dir, relative_path)
            code_str = FileUtils.read_file_as_string(file_path)
            code_lines = code_str.splitlines()
            self._file_cache[relative_path] = (code_str, code_lines)
        return self._file_cache[relative_path]

    def build_usage_mappings(self):
        """
        Retrieve all candidates callees and their callers for each repository.
        """
        method_start = time.time()

        # Phase 1: Extract function candidates
        candidates = set()
        extract_start = time.time()
        for task in tqdm.tqdm(self.tasks, total=len(self.tasks)):
            extractor = FunctionExtractor(task, shared_data=self.repo_data)
            new_candidates = extractor.extract_function_candidates()
            for nc in new_candidates:
                if nc['namespace'] not in candidates:
                    candidates.add(nc['namespace'])
        extract_time = time.time() - extract_start
        print(f'  [TIMING] Extracted {len(candidates)} candidates in {extract_time:.2f}s')

        # random remove candidate
        # candidates = remove_percentage(candidates, 50)

        # Phase 2: Get callers for each candidate
        usage_mappings = dict()
        callers_start = time.time()
        for candidate in candidates:
            usages = self.get_callers(candidate)
            usages = list(remove_percentage(set(usages), 50))
            if len(usages) > 0:
                usage_mappings[candidate] = usages
        callers_time = time.time() - callers_start
        print(f'  [TIMING] Retrieved callers for {len(usage_mappings)} candidates in {callers_time:.2f}s')

        # Phase 3: Save to disk
        save_start = time.time()
        mapping_path = os.path.join(Globals.MAPPING_DIR, f'{repo_name}.json')
        FileUtils.dump_json(usage_mappings, mapping_path)
        save_time = time.time() - save_start
        print(f'  [TIMING] Saved usage mappings to disk in {save_time:.2f}s')

        total_time = time.time() - method_start
        print(f'  [TIMING] Total build_usage_mappings time: {total_time:.2f}s')

        # time calculation
        return extract_time, callers_time

    def get_callers(self, callee_namespace):
        callers = []
        if callee_namespace in self.called_graph:
            callers.extend(self.called_graph[callee_namespace])
        return callers

    def retrieve_usage_examples(self, task, callee_namespace, callers, random_pick=False):
        # Cache task vector computation
        task_ns = task['namespace']
        if task_ns not in self._task_vector_cache:
            self._task_vector_cache[task_ns] = self.build_task_vector(task)
        task_vector = self._task_vector_cache[task_ns]

        usage_examples = []
        for c in callers:
            # prevent data leakage
            if c == task['namespace']:
                continue
            if c not in self.funcs:  # if the callers are classes or names
                continue
            caller_info = self.funcs[c]

            # OPTIMIZED: Use cached file content
            caller_code_str, caller_code_lines = self._get_file_content(caller_info['relative_path'])

            code = '\n'.join(caller_code_lines[caller_info['start_line_no']:caller_info['end_line_no']])
            ctx_end_line_no = caller_info['start_line_no']
            ctx_start_line_no = max(0, ctx_end_line_no - 20)
            ctx_window = '\n'.join(caller_code_lines[ctx_start_line_no:ctx_end_line_no])
            ctx_vector = self.tokenizer.tokenize(ctx_window)
            ctx_similarity = self.calculate_token_based_similarity(task_vector, ctx_vector)
            # store info
            tmp_info = copy.deepcopy(caller_info)  # keys: namespace, type, class, relpath, start/end line no
            tmp_info['callee'] = callee_namespace
            tmp_info['code'] = DataUtils.remove_multiline_comments(code)
            tmp_info['context_similarity'] = ctx_similarity
            usage_examples.append(tmp_info)
        if len(usage_examples) == 0:
            # print(f"{callee_namespace} does not found available usage in the repository.")
            return usage_examples
        if random_pick:
            random.shuffle(usage_examples)
            return usage_examples[:5]
        else:
            return sorted(usage_examples, key=lambda x: x['context_similarity'], reverse=True)[:5]

    def merge_usage_examples(self, task, random_pick=False):
        import time
        method_start = time.time()

        # Load repository data once using cache
        cache = RepositoryDataCache.get_instance()
        repo_data = cache.get_repository_data(self.repo_name, load_visitor=True,
                                               load_funcs=True, load_call_graph=True)

        # Step 1: Extract function candidates
        step_start = time.time()
        extractor = FunctionExtractor(task, shared_data=repo_data)
        candidates = extractor.extract_function_candidates()
        step_time = time.time() - step_start
        print(f"  [TIMING] extract_function_candidates: {step_time*1000:.2f}ms ({len(candidates)} candidates)")
        print(f"Found {len(candidates)} candidate functions for usage retrieval of {task['namespace']}.")

        # Step 2: Group candidates by caller set
        step_start = time.time()
        caller_set_to_candidates = {}
        for candidate in candidates:
            candidate_namespace = candidate['namespace']
            if candidate_namespace in self.mappings:
                # Create a hashable key from the caller set
                caller_set = tuple(sorted(self.mappings[candidate_namespace]))
                # speed up caller retrieval
                if len(caller_set) > 200:
                    continue
                if caller_set not in caller_set_to_candidates:
                    caller_set_to_candidates[caller_set] = []
                caller_set_to_candidates[caller_set].append(candidate)
        step_time = time.time() - step_start
        print(f"  [TIMING] group_candidates_by_caller_set: {step_time*1000:.2f}ms ({len(caller_set_to_candidates)} unique caller sets)")

        # Step 3: Process each unique caller set and retrieve usage examples
        step_start = time.time()
        usage_examples = []
        retrieval_count = 0
        for caller_set, candidate_group in caller_set_to_candidates.items():
            callers = list(caller_set)
            # print(f"Processing {len(callers)} callers for {len(candidate_group)} candidates.")

            # Retrieve usage examples once for this caller set
            examples = self.retrieve_usage_examples(task, candidate_group[0]['namespace'],
                                                    callers, random_pick)
            retrieval_count += 1
            # Replicate examples for all candidates in this group
            for candidate in candidate_group:
                for ex in examples:
                    ex_copy = copy.deepcopy(ex)
                    ex_copy['callee'] = candidate['namespace']
                    usage_examples.append(ex_copy)
        step_time = time.time() - step_start
        print(f"  [TIMING] retrieve_and_replicate_examples: {step_time*1000:.2f}ms ({retrieval_count} retrievals, {len(usage_examples)} total examples)")

        # Step 4: Deduplicate and merge callees
        step_start = time.time()
        k_dict = dict()
        for ex in usage_examples:
            if ex['namespace'] in k_dict:
                if ex['callee'] not in k_dict[ex['namespace']]['callee_namespace']:
                    k_dict[ex['namespace']]['callee_namespace'].append(ex['callee'])
            else:
                k_dict[ex['namespace']] = copy.deepcopy(ex)
                k_dict[ex['namespace']]['callee_namespace'] = [ex['callee']]
                del k_dict[ex['namespace']]['callee']
        step_time = time.time() - step_start
        print(f"  [TIMING] deduplicate_and_merge: {step_time*1000:.2f}ms ({len(k_dict)} deduplicated examples)")

        # Step 5: Sort and select top results
        step_start = time.time()
        if random_pick:
            tmp_examples = list(k_dict.values())
            random.shuffle(tmp_examples)
            result = tmp_examples[:10]
        else:
            result = sorted(k_dict.values(), key=lambda x: x['context_similarity'], reverse=True)[:10]
        step_time = time.time() - step_start
        print(f"  [TIMING] sort_and_select_top_k: {step_time*1000:.2f}ms ({len(result)} final results)")

        total_time = time.time() - method_start
        print(f"  [TIMING] Total merge_usage_examples: {total_time*1000:.2f}ms")

        return result

    def get_ground_truth_usage_examples(self, task, random_pick=False):
        usage_examples = []
        deps = itertools.chain.from_iterable(task['dependency'].values())
        for d in deps:
            if d in self.funcs:
                if d in self.mappings:
                    callers = self.mappings[d]
                    print(f"Found {len(callers)} callers for {d}.")
                    usage_examples.extend(self.retrieve_usage_examples(task, d, callers, random_pick))
                else:
                    print(f"=== Callers not found for {d}. ===")
        # remove duplicates and merge callees
        k_dict = dict()
        for ex in usage_examples:
            if ex['namespace'] in k_dict:
                if ex['callee'] not in k_dict[ex['namespace']]['callee_namespace']:
                    k_dict[ex['namespace']]['callee_namespace'].append(ex['callee'])
            else:
                k_dict[ex['namespace']] = copy.deepcopy(ex)
                k_dict[ex['namespace']]['callee_namespace'] = [ex['callee']]
                del k_dict[ex['namespace']]['callee']
        if random_pick:
            tmp_examples = list(k_dict.values())
            random.shuffle(tmp_examples)
            return tmp_examples[:10]
        else:
            return sorted(k_dict.values(), key=lambda x: x['context_similarity'], reverse=True)[:10]

    def build_task_vector(self, metadata):
        file_path = os.path.join(self.repo_base_dir, metadata['relative_path'])
        code = FileUtils.read_file_as_string(file_path)
        code_lines = code.splitlines()
        end_line_no = metadata['signature_position'][1]
        start_line_no = max(0, end_line_no - 20)
        task_window = '\n'.join(code_lines[start_line_no:end_line_no])
        return self.tokenizer.tokenize(task_window)

    @staticmethod
    def calculate_token_based_similarity(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return float(intersection) / union

    def clear_task_cache(self):
        """Clear task-specific caches to free memory."""
        self._task_vector_cache.clear()
        # Keep file cache as it's useful across tasks

    def clear_all_caches(self):
        """Clear all caches to free memory."""
        self._file_cache.clear()
        self._task_vector_cache.clear()


def build_item(task, example_list, keep_callees_only=False):
    repo_name = os.path.basename(task['repo_path'])
    if len(example_list) == 0:
        return None
    if keep_callees_only:
        callees = []
        for example in example_list:
            for c in example['callee_namespace']:
                if c not in callees:
                    callees.append(c)
        funcs = FileUtils.read_json(os.path.join(Globals.CANDIDATE_DIR, f"{repo_name}_funcs.json"))
        top_k_retrieved = [DataUtils.encapsulate_metadata_in_item(funcs[ns]) for ns in callees if ns in funcs]
    else:
        top_k_retrieved = [DataUtils.encapsulate_metadata_in_item(e) for e in example_list]
    return {
        'content': '',
        'metadata': task,
        'top_k_retrieved': top_k_retrieved
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_gt_deps', action='store_true', help='Use ground truth dependencies for retrieval')
    parser.add_argument('--only_callees', action='store_true', help='Use callees (instead of their callers) as retrieval results')
    parser.add_argument('--random_pick', action='store_true', help='Randomly select examples during retrieval')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    bench = 'deveval' #'codereval'
    # label = 'new' # 'full', 'codereval'

    tasks = FileUtils.read_jsonl(Globals.OUR_BENCHMARK_METADATA_PATH)
    # tasks = FileUtils.read_jsonl(Globals.FULL_BENCHMARK_METADATA_PATH)
    # tasks = FileUtils.read_jsonl(Globals.CODEREVAL_BENCHMARK_METADATA_PATH)

    # group tasks by repos
    repo_to_task_mappings = dict()
    for t in tasks:
        repo = t['repo_path']
        repo_to_task_mappings.setdefault(repo, [])
        repo_to_task_mappings[repo].append(t)

    time_dict = {
        'DGC': dict(),
        'LCE': dict(),
        'UER': dict()
    }

    # Create tokenizer once globally
    tokenizer = Tokenizer('cl100k_base')

    selected_example_path = os.path.join(Globals.LOGICODER_PROMPT_DIR, 'selected_examples.json') # f'selected_examples-{label}.json'
    if not os.path.exists(selected_example_path):
        selected_examples = dict()
        # Process each repository
        processing_start = time.time()
        for repo_path, task_list in repo_to_task_mappings.items():
            repo_name = repo_path.split('/')[-1]
            print(f'\nProcessing repository: {repo_name} ({len(task_list)} tasks)')

            print('\n' + '=' * 60)
            print('PHASE 1: Building call graphs')
            print('=' * 60)
            phase1_start = time.time()
            # Build call graphs
            CallGraphBuilder(bench=bench).build_call_graphs(repo_path)
            phase1_time = time.time() - phase1_start
            print(f'[TIMING] Build graphs for {len(task_list)} tasks in {phase1_time:.2f}s')
            time_dict['DGC'][repo_name] = phase1_time

            print('=' * 60)
            print('PHASE 2: Selecting usage examples for each task')
            print('=' * 60)
            phase2_start = time.time()
            # Pre-load repository data cache for all repositories
            cache = RepositoryDataCache.get_instance()
            print('\nPre-loading repository data into cache...')
            preload_start = time.time()
            cache.get_repository_data(repo_name, load_visitor=True, load_funcs=True,
                                      load_call_graph=True, load_called_graph=True,
                                      load_mappings=True)
            preload_time = time.time() - preload_start
            print(f'[TIMING] Pre-loaded {len(repo_to_task_mappings)} repositories in {preload_time:.2f}s\n')

            # Initialize retriever
            retriever = UsageExampleRetriever(repo_name, task_list, tokenizer=tokenizer, bench=bench)
            LCE, UER1 = retriever.build_usage_mappings()

            for m in tqdm.tqdm(task_list, total=len(task_list)):
                if args.with_gt_deps:
                    usage_examples = retriever.get_ground_truth_usage_examples(m, random_pick=args.random_pick)
                else:
                    usage_examples = retriever.merge_usage_examples(m, random_pick=args.random_pick)
                selected_examples[m['namespace']] = usage_examples

            UER2 = time.time() - phase2_start
            print(f'[TIMING] Selected examples for {len(task_list)} tasks in {UER1 + UER2:.2f}s')
            time_dict['LCE'][repo_name] = LCE
            time_dict['UER'][repo_name] = UER1 + UER2

            repo_time = time.time() - phase1_start
            print(f'[TIMING] Total time for {repo_name}: {repo_time:.2f}s\n')

        FileUtils.dump_json(selected_examples, selected_example_path)
        total_repo_time = time.time() - processing_start
        print('='*60)
        print(f'[TIMING] Total time for all repositories: {total_repo_time:.2f}s')
        print('='*60 + '\n')
        # FileUtils.dump_json(time_dict, os.path.join(Globals.ROOT_DIR, 'logs', 'latency', 'logicoder.json'))
    else:
        load_start = time.time()
        selected_examples = FileUtils.read_json(selected_example_path)
        load_time = time.time() - load_start
        print(f'[TIMING] Loaded cached selected examples in {load_time:.2f}s')
    print('='*60 + '\n')

    reranked_example_path = os.path.join(Globals.LOGICODER_PROMPT_DIR, f'reranked_examples.json')
    print('='*60)
    print('PHASE 3: Reranking examples')
    print('='*60)
    if not os.path.exists(reranked_example_path):
        phase3_start = time.time()
        main(bench=bench)
        reranked_examples = rerank(selected_examples, bench=bench)
        phase3_time = time.time() - phase3_start
        print(f'[TIMING] Reranked {len(reranked_examples)} examples in {phase3_time:.2f}s')
    else:
        load_start = time.time()
        reranked_examples = FileUtils.read_json(reranked_example_path)
        load_time = time.time() - load_start
        print(f'[TIMING] Loaded cached reranked examples in {load_time:.2f}s')
    print('='*60 + '\n')

    # build prompts
    prompt_lines = []
    for task in tqdm.tqdm(tasks, total=len(tasks)):
        usage_examples = reranked_examples[task['namespace']]
        # usage_examples = selected_examples[task['namespace']]
        item = build_item(task, usage_examples, args.only_callees)
        if item is None:
            continue
        prompt_line = RAGPromptBuilder(item, bench=bench).build_prompt_line()
        prompt_lines.append(prompt_line)
    print(f"Build {len(prompt_lines)} prompts from rag pipeline.")
    # add direct generation prompts
    finished = [l['query_window']['metadata']['namespace'] for l in prompt_lines]
    # print(finished)
    tmp_path = Globals.REPO_BASE_DIR if bench == 'deveval' else Globals.REPO_BASE_CODEREVAL_DIR
    for task in tasks:
        repo_path = os.path.join(tmp_path, task['repo_path'])
        if task['namespace'] not in finished:
            prompt_line = DirectPromptBuilder(task, None, True, bench=bench).build_prompt_line(mode='rag')
            prompt_lines.append(prompt_line)
    print(f"Build {len(prompt_lines)} prompts in total.")

    prompt_path = os.path.join(Globals.LOGICODER_PROMPT_DIR, f'logicoder.jsonl')
    FileUtils.dump_jsonl(prompt_lines, prompt_path)
