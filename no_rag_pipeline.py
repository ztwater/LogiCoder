import os

from build_prompt import DirectPromptBuilder
from globals import Globals
from myutils import FileUtils


class NoRAGRunner:
    def __init__(self, repos, benchmark_metadata, bench='deveval'):
        self.repos = repos  # the absolute paths of repos
        self.benchmark_metadata = benchmark_metadata
        self.skeletons = dict()
        self.bench = bench

    def build_prompts_for_no_rag_generation(self):
        prompt_lines = []
        for metadata in self.benchmark_metadata:
            prompt_line = DirectPromptBuilder(metadata, context_length=None, infile_context=False, bench=self.bench).build_prompt_line(mode='direct')
            prompt_lines.append(prompt_line)
        print(f"Build {len(prompt_lines)} prompts in total.")
        prompt_path = os.path.join(Globals.NO_RAG_PROMPT_DIR, f'direct-{self.bench}.jsonl')
        FileUtils.dump_jsonl(prompt_lines, prompt_path)

    def build_prompts_for_infile_context_generation(self):
        prompt_lines = []
        for metadata in self.benchmark_metadata:
            prompt_line = DirectPromptBuilder(metadata, context_length=None, infile_context=True, bench=self.bench).build_prompt_line(mode='direct')
            prompt_lines.append(prompt_line)
        print(f"Build {len(prompt_lines)} prompts in total.")
        prompt_path = os.path.join(Globals.NO_RAG_PROMPT_DIR, f'infile-{self.bench}.jsonl')
        FileUtils.dump_jsonl(prompt_lines, prompt_path)


def no_rag_workflow(repos, metadata, bench='deveval'):
    runner = NoRAGRunner(repos, metadata, bench=bench)
    # runner.build_prompts_for_no_rag_generation()
    runner.build_prompts_for_infile_context_generation()


if __name__ == '__main__':
    bench = 'deveval'  # codereval  # deveval

    repos = FileUtils.get_repositories(bench)
    print(f"Target repos ({len(repos)} in total):\n{repos}")

    if bench == 'deveval':
        # updated_metadata = FileUtils.read_jsonl(Globals.OUR_BENCHMARK_METADATA_PATH)
        updated_metadata = FileUtils.read_jsonl(Globals.FULL_BENCHMARK_METADATA_PATH)
    else:
        updated_metadata = FileUtils.read_jsonl(Globals.CODEREVAL_BENCHMARK_METADATA_PATH)

    # baselines
    no_rag_workflow(repos, updated_metadata, bench=bench)

    from cot_pipeline import cot_and_code_workflow, self_planning_workflow
    # cot_and_code_workflow(updated_metadata, bench=bench)
    # self_planning_workflow(updated_metadata, bench=bench)
