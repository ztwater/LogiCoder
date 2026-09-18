import copy
import os

import numpy as np
import torch
import tqdm

from globals import Globals
from myutils import FileUtils

def get_similarity(l1, l2, is_dense):
    if is_dense:
        embedding1 = l1['data'][0]['embedding']
        embedding2 = l2['data'][0]['embedding']
        norm_embedding1 = torch.nn.functional.normalize(embedding1, p=2, dim=1)
        norm_embedding2 = torch.nn.functional.normalize(embedding2, p=2, dim=1)
        return torch.einsum("ac,bc->ab", norm_embedding1, norm_embedding2).item()
    else:
        list1 = np.array(l1['data'][0]['embedding'])
        list2 = np.array(l2['data'][0]['embedding'])
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return float(intersection) / union

def retrieve(query_line, func_lines, is_dense):
    ns_q = query_line['metadata']['namespace']
    label = 'sem' if is_dense else 'ctx'
    target_path = os.path.join(Globals.RETRIEVAL_DIR, 'sim', f"{ns_q}_{label}.pkl")
    if os.path.exists(target_path):
        return
    else:
        print(f"Search task {ns_q}.")
        sims = dict()
        for func_line in func_lines:
            ns_f = func_line['metadata']['namespace']
            similarity = get_similarity(query_line, func_line, is_dense)
            sims[ns_f] = similarity
        sorted_sims = {k: v for k, v in sorted(sims.items(), key=lambda x: x[1], reverse=True)}
        FileUtils.dump_pickle(sorted_sims, target_path)

def main(bench='deveval'):
    if bench == 'deveval':
        # tasks = FileUtils.read_jsonl(Globals.OUR_BENCHMARK_METADATA_PATH)
        tasks = FileUtils.read_jsonl(Globals.FULL_BENCHMARK_METADATA_PATH)
    else:
        tasks = FileUtils.read_jsonl(Globals.CODEREVAL_BENCHMARK_METADATA_PATH)
    s_vec_name = 'bow'
    d_vec_name = 'unixcoder'
    window_size = 20

    repos = set([t['repo_path'] for t in tasks])

    for repo in repos:
        print(f"Working on {repo}")
        repo_name = repo.split('/')[-1]
        # req_vector_path = os.path.join(Globals.KNOWLEDGE_BASE_DIR, 'vector', 'req',
        #                                 f'{repo_name}_{d_vec_name}.pkl')
        # func_vector_path = os.path.join(Globals.KNOWLEDGE_BASE_DIR, 'vector', 'func',
        #                                 f'{repo_name}_funcs_{d_vec_name}.pkl')
        task_vector_path = os.path.join(Globals.KNOWLEDGE_BASE_DIR, 'vector', 'benchmark',
                                             f'{repo_name}_ws50_{s_vec_name}.pkl')
        func_vector_path = os.path.join(Globals.KNOWLEDGE_BASE_DIR, 'vector', 'func',
                                        f'{repo_name}_funcs_{s_vec_name}.pkl')
        req_lines = FileUtils.load_pickle(task_vector_path)
        func_lines = FileUtils.load_pickle(func_vector_path)
        for req_line in tqdm.tqdm(req_lines, total=len(req_lines)):
            # retrieve(req_line, func_lines, is_dense=True)
            retrieve(req_line, func_lines, is_dense=False)

        task_ctx_vector_path = os.path.join(Globals.KNOWLEDGE_BASE_DIR, 'vector', 'benchmark',
                                            f'{repo_name}_ws{window_size}_{s_vec_name}.pkl')
        func_ctx_vector_path = os.path.join(Globals.KNOWLEDGE_BASE_DIR, 'vector', 'func',
                                            f'{repo_name}_funcs-ctx_{s_vec_name}.pkl')
        task_ctx_lines = FileUtils.load_pickle(task_ctx_vector_path)
        func_ctx_lines = FileUtils.load_pickle(func_ctx_vector_path)
        for task_ctx_line in tqdm.tqdm(task_ctx_lines, total=len(task_ctx_lines)):
            retrieve(task_ctx_line, func_ctx_lines, is_dense=False)


def rerank(selected_examples, bench='deveval'):
    if bench == 'deveval':
        # tasks = FileUtils.read_jsonl(Globals.OUR_BENCHMARK_METADATA_PATH)
        tasks = FileUtils.read_jsonl(Globals.FULL_BENCHMARK_METADATA_PATH)
    else:
        tasks = FileUtils.read_jsonl(Globals.CODEREVAL_BENCHMARK_METADATA_PATH)
    reranked_examples = dict()

    task_to_repo_mappings = dict()
    for t in tasks:
        task_to_repo_mappings[t['namespace']] = t['repo_path'].split('/')[-1]

    for namespace in tqdm.tqdm(selected_examples):
        sem_sims = FileUtils.load_pickle(os.path.join(Globals.RETRIEVAL_DIR, 'sim', f"{namespace}_sem.pkl"))
        ctx_sims = FileUtils.load_pickle(os.path.join(Globals.RETRIEVAL_DIR, 'sim', f"{namespace}_ctx.pkl"))
        ex_list = selected_examples[namespace]
        top_10_log = [ex['namespace'] for ex in ex_list]
        top_10_sem = [ex for ex in sem_sims if ex != namespace][:10]  # prevent the data leakage from ground truth

        sim_order = dict()
        ctx_order = dict()
        for ns in top_10_log + top_10_sem:
            if ns in sim_order:
                continue
            sim_order[ns] = sem_sims[ns]
            ctx_order[ns] = ctx_sims[ns]
        sim_order = sorted(sim_order.items(), key=lambda x: x[1], reverse=True)
        ctx_order = sorted(ctx_order.items(), key=lambda x: x[1], reverse=True)
        merged = dict()
        for idx, item in enumerate(sim_order):
            ns = item[0]
            # merged[ns] = item[1]  # baseline ScoreAdd
            merged[ns] = 1 / (60 + idx + 1)  # k = 60
        for idx, item in enumerate(ctx_order):
            ns = item[0]
            # merged[ns] += item[1]
            merged[ns] += 1 / (60 + idx + 1)
        merged = sorted(merged.items(), key=lambda x: x[1], reverse=True)

        repo_name = task_to_repo_mappings[namespace]
        func_path = os.path.join(Globals.CANDIDATE_DIR, f'{repo_name}_funcs.json')
        funcs = FileUtils.read_json(func_path)
        reranked_items = []
        for item in merged:
            try:
                it = copy.deepcopy(funcs[item[0]])
            except KeyError as e:
                print(e)
                continue
            it['sem_sim'] = sem_sims[item[0]]
            it['ctx_sim'] = ctx_sims[item[0]]
            reranked_items.append(it)
        reranked_examples[namespace] = reranked_items
    reranked_example_path = os.path.join(Globals.LOGICODER_PROMPT_DIR, 'reranked_examples.json')
    FileUtils.dump_json(reranked_examples, reranked_example_path)
    return reranked_examples

if __name__ == '__main__':
    selected_example_path = os.path.join(Globals.LOGICODER_PROMPT_DIR, 'selected_examples.json')
    selected_examples = FileUtils.read_json(selected_example_path)
    namespace = 'boto.ec2.volume.Volume.update'
    sem_sims = FileUtils.load_pickle(os.path.join(Globals.RETRIEVAL_DIR, 'sim', f"{namespace}_sem.pkl"))
    ctx_sims = FileUtils.load_pickle(os.path.join(Globals.RETRIEVAL_DIR, 'sim', f"{namespace}_ctx.pkl"))
    ex_list = selected_examples[namespace]
    top_10_log = [ex['namespace'] for ex in ex_list]
    top_10_sem = [ex for ex in sem_sims if ex != namespace][:10]  # prevent the data leakage from ground truth

    sim_order = dict()
    ctx_order = dict()
    for ns in top_10_log + top_10_sem:
        if ns in sim_order:
            continue
        sim_order[ns] = sem_sims[ns]
        ctx_order[ns] = ctx_sims[ns]
    sim_order = sorted(sim_order.items(), key=lambda x: x[1], reverse=True)
    ctx_order = sorted(ctx_order.items(), key=lambda x: x[1], reverse=True)
    merged = dict()
    for idx, item in enumerate(sim_order):
        ns = item[0]
        merged[ns] = 1 / (60 + idx + 1)  # k = 60

    for idx, item in enumerate(ctx_order):
        ns = item[0]
        merged[ns] += 1 / (60 + idx + 1)
    merged = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    print(sim_order)
    print(ctx_order)
    print(merged)
