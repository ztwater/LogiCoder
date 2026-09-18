from concurrent.futures import as_completed, ProcessPoolExecutor
import os
import tqdm

from build_prompt import RepoCoderPromptBuilder
from build_window import RepoWindowBuilder, BenchmarkWindowBuilder, PredictionWindowBuilder, GroundTruthWindowBuilder
from build_vector import BagOfWordVectorizer, UniXCoderVectorizer
from retriever import RepoCoderSparseRetriever, RepoCoderDenseRetriever

from globals import Globals
from myutils import FileUtils, PathUtils, Tokenizer


class RepoCoderRunner:
    def __init__(self, repos, window_size, slice_size, vectorizer, top_k, prediction_path=None):
        self.repos = repos
        self.window_size = window_size
        self.slice_size = slice_size
        self.tokenizer = Tokenizer()
        self.vectorizer = vectorizer
        self.top_k = top_k
        self.prediction_path = prediction_path

    def run_repo_vectorization(self):
        for repo in self.repos:
            RepoWindowBuilder(repo, self.window_size, self.slice_size).build_windows()
            repo_window_path = PathUtils.get_repo_window_path(repo, self.window_size, self.slice_size)
            if self.vectorizer.get_name() == 'bow':
                self.vectorizer(repo_window_path, self.tokenizer).build()
            elif self.vectorizer.get_name() == 'unixcoder':
                self.vectorizer(repo_window_path).build()

    def run_benchmark_vectorization(self, metadata, bench='deveval'):
        for repo in self.repos:  # store tasks by repo
            BenchmarkWindowBuilder(metadata, repo, self.window_size).build_windows(bench=bench)
            benchmark_window_path = PathUtils.get_benchmark_window_path(repo, self.window_size)
            if self.vectorizer.get_name() == 'bow':
                self.vectorizer(benchmark_window_path, self.tokenizer).build()
            elif self.vectorizer.get_name() == 'unixcoder':
                self.vectorizer(benchmark_window_path).build()

    def run_ground_truth_vectorization(self, metadata):
        for repo in self.repos:  # store tasks by repo
            GroundTruthWindowBuilder(metadata, repo, self.window_size).build_windows()
            ground_truth_window_path = PathUtils.get_ground_truth_window_path(repo, self.window_size)
            vectorizer(ground_truth_window_path, self.tokenizer).build()

    def set_prediction_path(self, prediction_path):
        self.prediction_path = prediction_path

    def run_prediction_vectorization(self, bench='deveval'):
        for repo in self.repos:
            PredictionWindowBuilder(self.prediction_path, repo, self.window_size, self.slice_size).build_windows(bench=bench)
            prediction_window_path = PathUtils.get_prediction_window_path(repo, self.prediction_path)
            if self.vectorizer.get_name() == 'bow':
                self.vectorizer(prediction_window_path, self.tokenizer).build()
            elif self.vectorizer.get_name() == 'unixcoder':
                self.vectorizer(prediction_window_path).build()


    def run_retrieval_sparse(self, turn):
        retrievers = []
        for repo in self.repos:
            query_window_path = self.get_query_window_path(repo, turn)
            query_vector_path = PathUtils.get_vector_path(query_window_path, self.vectorizer.get_name())
            repo_window_path = PathUtils.get_repo_window_path(repo, self.window_size, self.slice_size)
            repo_vector_path = PathUtils.get_vector_path(repo_window_path, self.vectorizer.get_name())
            query_lines = FileUtils.load_pickle(query_vector_path)
            repo_lines = FileUtils.load_pickle(repo_vector_path)
            retrieval_path = PathUtils.get_retrieval_path(query_vector_path, self.top_k)
            retriever = RepoCoderSparseRetriever(repo, query_lines, repo_lines, self.top_k, retrieval_path)
            retrievers.append(retriever)

        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            futures = {executor.submit(retriever.retrieve) for retriever in retrievers}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                future.result()

    def run_retrieval_dense(self, turn):
        for repo in self.repos:
            query_window_path = self.get_query_window_path(repo, turn)
            query_vector_path = PathUtils.get_vector_path(query_window_path, self.vectorizer.get_name())
            repo_window_path = PathUtils.get_repo_window_path(repo, self.window_size, self.slice_size)
            repo_vector_path = PathUtils.get_vector_path(repo_window_path, self.vectorizer.get_name())
            query_lines = FileUtils.load_pickle(query_vector_path)
            repo_lines = FileUtils.load_pickle(repo_vector_path)
            retrieval_path = PathUtils.get_retrieval_path(query_vector_path, self.top_k)
            retriever = RepoCoderDenseRetriever(repo, query_lines, repo_lines, self.top_k, retrieval_path)
            retriever.retrieve()

    def run_prompt_builder(self, turn, need_shift=False, bench='deveval'):
        prompt_lines = []
        for repo in self.repos:
            query_window_path = self.get_query_window_path(repo, turn)
            query_vector_path = PathUtils.get_vector_path(query_window_path, self.vectorizer.get_name())
            retrieval_path = PathUtils.get_retrieval_path(query_vector_path, self.top_k)
            retrieved_results = FileUtils.load_pickle(retrieval_path)
            for retrieved_result in retrieved_results:
                prompt_line = RepoCoderPromptBuilder(retrieved_result, need_shift=need_shift, bench=bench).build_prompt_line()
                prompt_lines.append(prompt_line)

        prompt_path = os.path.join(Globals.REPOCODER_PROMPT_DIR,
                                   f'ws{window_size}_ss{slice_size}_{vectorizer.get_name()}_{turn}-full.jsonl')
        FileUtils.dump_jsonl(prompt_lines, prompt_path)

    def get_query_window_path(self, repo, turn):
        if turn == 'first':
            return PathUtils.get_benchmark_window_path(repo, self.window_size)
        elif turn == 'gt':
            return PathUtils.get_ground_truth_window_path(repo, self.window_size)
        else:
            prediction_name = os.path.basename(self.prediction_path)
            return PathUtils.get_prediction_window_path(repo, prediction_name)


if __name__ == '__main__':
    bench = 'deveval'  # 'codereval'
    repos = FileUtils.get_repositories(bench)
    window_size = 50
    slice_size = 5
    vectorizer = BagOfWordVectorizer
    # vectorizer = UniXCoderVectorizer
    top_k = 20

    if slice_size > window_size:
        raise Exception('The slice size should be less or equal to the window size.')

    # updated_metadata = FileUtils.read_jsonl(Globals.OUR_BENCHMARK_METADATA_PATH)
    updated_metadata = FileUtils.read_jsonl(Globals.FULL_BENCHMARK_METADATA_PATH)
    # updated_metadata = FileUtils.read_jsonl(Globals.CODEREVAL_BENCHMARK_METADATA_PATH)

    runner = RepoCoderRunner(repos, window_size, slice_size, vectorizer, top_k)
    turn = 'second'  # 'first', 'second', 'third'

    # Step 1: build windows for each repo
    # runner.run_repo_vectorization()
    # Step 2: build windows for queries in benchmark
    # runner.run_benchmark_vectorization(updated_metadata, bench)
    # runner.run_ground_truth_vectorization(metadata)
    # Step 3: retrieve similar code
    # runner.run_retrieval_sparse(turn)
    # runner.run_retrieval_dense(turn)
    # Step 4: build prompt
    # runner.run_prompt_builder(turn, need_shift=True, bench=bench)
    # Step 5: run LLM inference
    # run inference.py with args

    # iterative running
    prediction_path = os.path.join(Globals.PREDICTION_DIR, 'completion_repocoder_ws50_ss5_bow_first-full_deepseek-v3-250324_T0_N1.jsonl')
    runner.set_prediction_path(prediction_path)
    # Step 6: build prediction window
    # runner.run_prediction_vectorization(bench)
    # Step 7: retrieve similar code
    # runner.run_retrieval_sparse(turn)
    # runner.run_retrieval_dense(turn)
    # Step 8: build prompt
    runner.run_prompt_builder(turn, need_shift=True, bench=bench)  # do not shift for the second iteration
