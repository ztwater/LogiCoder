import copy
import itertools
import os
import tqdm

from myutils import FileUtils
from globals import Globals


class Evaluator:
    def __init__(self, repos, benchmark_metadata):
        self.repos = repos
        self.benchmark_metadata = benchmark_metadata

    def build_dependency_map(self):
        dep_map_path = os.path.join(Globals.ROOT_DIR, 'func_dep_map.json')
        try:
            return FileUtils.read_json(dep_map_path)
        except FileNotFoundError:
            dep_map = dict()
            for metadata in self.benchmark_metadata:
                namespace = metadata['namespace']
                repo_name = os.path.basename(metadata['repo_path'])
                dep_map[namespace] = []
                func_path = os.path.join(Globals.UNIT_DIR, f'{repo_name}_funcs.json')
                funcs = FileUtils.read_json(func_path)
                for dep in itertools.chain.from_iterable(metadata['dependency'].values()):
                    if dep in funcs:
                        dep_map[namespace].append(dep)
            FileUtils.dump_json(dep_map, dep_map_path)
            return dep_map

    def evaluate_knowledge_extraction(self, dep_map):
        from extract_function import FunctionExtractor
        coverage = []
        for metadata in tqdm.tqdm(self.benchmark_metadata, total=len(self.benchmark_metadata)):
            extractor = FunctionExtractor(metadata)
            candidates = extractor.extract_function_candidates()
            c_set = set([c['namespace'] for c in candidates])
            deps = set(dep_map[metadata['namespace']])
            recall = self.recall_k(c_set, deps)
            coverage.append(recall)
        print(f"recall: {sum(coverage)/len(coverage)}")

    def evaluate_candidate_func(self, dep_map, name_string):
        evaluations = []
        for metadata in tqdm.tqdm(self.benchmark_metadata, total=len(self.benchmark_metadata)):

            example_path = os.path.join(Globals.LOGICODER_PROMPT_DIR, f'selected_examples{name_string}.json')
            examples = FileUtils.read_json(example_path)

            callee_set = set()
            callee_list = []
            for ex in examples[metadata['namespace']]:
                new_callees = ex['callee_namespace']
                callee_set.update(new_callees)
                s = copy.deepcopy(callee_set)
                callee_list.append(s)

            if len(callee_list) == 0:
                callee_list = [[], [], [], [], [], [], [], [], [], []]
            else:
                for i in range(10 - len(callee_list)):
                    callee_list.append(callee_list[-1])
            # print([len(v) for v in callee_list])

            deps = set(dep_map[metadata['namespace']])

            result = dict()
            if len(deps) > 0:
                # calculate the precision_k and recall_k
                for k in [1, 3, 5, 10]:
                    top_k_set = set(callee_list[k-1])
                    recall_k = self.recall_k(top_k_set, deps)
                    result[f'recall_{k}'] = recall_k

            evaluations.append(result)
        # merge evaluation results
        from collections import defaultdict
        merged_dict = defaultdict(list)
        for d in evaluations:
            for key, value in d.items():
                merged_dict[key].append(value)
        print([(k, round((sum(v) / len(v)), 4)) for k, v in merged_dict.items()])


    @staticmethod
    def recall_k(pred_set, true_set):
        overlap = pred_set.intersection(true_set)
        return len(overlap) / len(true_set)

    @staticmethod
    def precision_k(pred_set, true_set):
        overlap = pred_set.intersection(true_set)
        return len(overlap) / len(pred_set)


if __name__ == '__main__':
    repos = FileUtils.get_repositories()

    # only conduct experiments on tasks with cross file dependencies
    benchmark_metadata = FileUtils.read_jsonl(Globals.OUR_BENCHMARK_METADATA_PATH)

    eval = Evaluator(repos, benchmark_metadata)
    dep_map = eval.build_dependency_map()

    eval.evaluate_knowledge_extraction(dep_map)

    eval.evaluate_candidate_func(dep_map, '')
    # eval.evaluate_candidate_func(dep_map, '_no_import')
    # eval.evaluate_candidate_func(dep_map, '_no_assoc')
    # eval.evaluate_candidate_func(dep_map, '_no_infile')