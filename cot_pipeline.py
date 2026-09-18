import json
import os

from build_prompt import CoTPromptBuilder, SelfPlanningPromptBuilder
from myutils import FileUtils
from globals import Globals

class CoTRunner:
    def __init__(self, metadata, bench='deveval'):
        self.metadata = metadata
        self.bench = bench

    def build_cot_prompts(self, include_code=False):
        num_of_demos = 3
        if include_code:
            demo_path = os.path.join(Globals.DEMO_DIR, 'cot_and_code_demo.txt')
            prompt_path = os.path.join(Globals.COT_PROMPT_DIR, f'plan_and_code-{self.bench}.jsonl')
        else:
            demo_path = os.path.join(Globals.DEMO_DIR, 'cot_demo.txt')
            prompt_path = os.path.join(Globals.COT_PROMPT_DIR, f'plan-{self.bench}.jsonl')
        prompts = []
        for metadata in self.metadata:
            prompt_line = CoTPromptBuilder(metadata, num_of_demos, demo_path, include_code, bench=self.bench).build_prompt_line()
            prompts.append(prompt_line)
        FileUtils.dump_jsonl(prompts, prompt_path)

    def extract_cot_results(self):
        cot_path = os.path.join(Globals.PREDICTION_DIR, 'completion_cot_plan_deepseek-chat_T0_N1.jsonl')
        cot_lines = FileUtils.read_jsonl(cot_path)
        cot_windows = []
        for line in cot_lines:
            for pred in [sample['text'] for sample in line['predictions']]:
                plan = json.loads(pred)
                cot_windows.append({
                    'content': plan,
                    'metadata': line['metadata']
                })
        print(f'Build {len(cot_windows)} cot windows in total.')
        return cot_windows


def cot_and_code_workflow(metadata, bench='deveval'):
    CoTRunner(metadata, bench=bench).build_cot_prompts(include_code=True)


def self_planning_workflow(metadata, bench='deveval'):
    cot_runner = CoTRunner(metadata, bench=bench)
    cot_runner.build_cot_prompts()
    cot_windows = cot_runner.extract_cot_results()
    plan_dict = dict()
    for line in cot_windows:
        plan_dict[line['metadata']['namespace']] = line['content']
    prompt_lines = []
    for m in metadata:
        import textwrap
        file_path = os.path.join(Globals.REPO_BASE_DIR, m['relative_path'])
        code = FileUtils.read_file_as_string(file_path)
        code_lines = code.splitlines()
        start_line_no, end_line_no = m['signature_position']
        signature_lines = code_lines[start_line_no:end_line_no]
        functionality = m['requirement']['Functionality']
        arguments = m['requirement']['Arguments']
        plan = '\n'.join([f"{i+1}. {step}" for i, step in enumerate(plan_dict[m['namespace']])])
        requirement = (f"\"\"\"\n{functionality}\nInput-Output Arguments:\n{arguments}\n"
                       f"Plan:\n{plan}\"\"\"")
        indent = m['indent']
        requirement_with_indent = textwrap.indent(requirement, ' ' * indent)
        signature = '```python\n' + '\n'.join(signature_lines) + '\n' + requirement_with_indent + '\n```\n'
        prompt_line = SelfPlanningPromptBuilder(m, signature, None, True).build_prompt_line()
        prompt_lines.append(prompt_line)
    print(f"Build {len(prompt_lines)} prompts in total.")
    prompt_path = os.path.join(Globals.COT_PROMPT_DIR, 'self_planning.jsonl')
    FileUtils.dump_jsonl(prompt_lines, prompt_path)
