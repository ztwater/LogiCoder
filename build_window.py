from collections import defaultdict
import os

from globals import Globals
from myutils import DataUtils, FileUtils, PathUtils


class RepoWindowBuilder:
    def __init__(self, repo, window_size, slice_size):
        self.repo = repo
        self.window_size = window_size
        self.slice_size = slice_size
        self.slice_step = window_size // slice_size
        self.source_code_files = FileUtils.find_py_files(repo)  # FileUtils.iterate_repository(repo)

    def build_windows_for_file(self, relative_path, code):
        windows = []
        code_lines = code.splitlines()
        for start_line_no in range(0, len(code_lines) - self.window_size + self.slice_step, self.slice_step):  # line_no starts from 0
            end_line_no = min(len(code_lines), start_line_no + self.window_size)
            lines_in_window = [i for i in code_lines[start_line_no:end_line_no]]  # create a new copy of lines
            if not lines_in_window:  # all empty lines
                continue
            windows.append({
                'content': '\n'.join(lines_in_window),
                'metadata': {
                    'repo_path': self.repo,
                    'relative_path': relative_path,
                    'start_line_no': start_line_no,
                    'end_line_no': end_line_no,
                    'window_size': self.window_size,
                    'slice_size': self.slice_size,
                }
            })
        return windows

    @staticmethod
    def merge_windows_with_same_context(windows):
        merged_dict = defaultdict(list)
        for code_window in windows:
            content = code_window['content']
            metadata = code_window['metadata']
            merged_dict[content].append(metadata)
        merged_windows = []
        for content, metadata_list in merged_dict.items():
            merged_windows.append({
                'content': content,
                'metadata': metadata_list
            })
        return merged_windows

    def build_windows(self, bench='deveval'):
        repo_base_dir = Globals.REPO_BASE_DIR if bench == 'deveval' else Globals.REPO_BASE_CODEREVAL_DIR
        windows_in_repo = []
        for code_file in self.source_code_files:
            relative_path = os.path.relpath(code_file, repo_base_dir)
            code = FileUtils.read_file_as_string(code_file)
            windows_in_repo += self.build_windows_for_file(relative_path, code)
        # merged_code_windows = windows_in_repo
        merged_code_windows = RepoWindowBuilder.merge_windows_with_same_context(windows_in_repo)
        print(f'Build {len(merged_code_windows)} repo windows for {os.path.basename(self.repo)} with window size {self.window_size} and slice {self.slice_size}.')
        output_path = PathUtils.get_repo_window_path(self.repo, self.window_size, self.slice_size)
        FileUtils.dump_pickle(merged_code_windows, output_path)

class BenchmarkWindowBuilder:
    def __init__(self, metadata, repo, window_size):
        self.tasks = metadata
        self.repo = repo
        self.window_size = window_size

    # def build_windows(self):
    #     windows_in_benchmark = []
    #     for task in self.tasks:
    #         repo_path = os.path.join(Globals.REPO_BASE_DIR, task['project_path'])
    #         if self.repo != repo_path:
    #             continue
    #         file_path = os.path.join(Globals.REPO_BASE_DIR, task['completion_path'])
    #         code = FileUtils.read_file_as_string(file_path)
    #         code_lines = code.splitlines()
    #         line_no = task['body_position'][0] - 1  # line_no starts from 0
    #         start_line_no = max(0, line_no - self.window_size)
    #         lines_in_window = code_lines[start_line_no:line_no]  # do not consider code after the method
    #         windows_in_benchmark.append({
    #             'content': '\n'.join(lines_in_window),
    #             'metadata': {
    #                 'namespace': task['namespace'],
    #                 'type': task['type'],  # 'function'
    #                 'repo_path': repo_path,
    #                 'relative_path': task['completion_path'],
    #                 'start_line_no': start_line_no,
    #                 'end_line_no': line_no,
    #                 'window_size': self.window_size,
    #                 'signature_position': [task['signature_position'][0] - 1, task['signature_position'][1]],  # deveval starts from 1
    #                 'body_position': [task['body_position'][0] - 1, task['body_position'][1]],
    #                 'dependency': task['dependency'],
    #                 'requirement': task['requirement'],
    #                 'tests': task['tests'],
    #                 'indent': task['indent']
    #             }
    #         })
    #     print(f'Build {len(windows_in_benchmark)} benchmark windows for {os.path.basename(self.repo)} with window size {self.window_size}.')
    #     output_path = PathUtils.get_benchmark_window_path(self.repo, self.window_size)
    #     FileUtils.dump_pickle(windows_in_benchmark, output_path)

    def build_windows(self, bench='deveval'):
        repo_base_dir = Globals.REPO_BASE_DIR if bench == 'deveval' else Globals.REPO_BASE_CODEREVAL_DIR
        windows_in_benchmark = []
        for task in self.tasks:
            repo_path = os.path.join(repo_base_dir, task['repo_path'])
            if self.repo != repo_path:
                continue
            file_path = os.path.join(repo_base_dir, task['relative_path'])
            code = FileUtils.read_file_as_string(file_path)
            code_lines = code.splitlines()
            line_no = task['signature_position'][1]
            start_line_no = max(0, line_no - self.window_size)
            lines_in_window = code_lines[start_line_no:line_no]  # do not consider code after the method
            # print(task)
            windows_in_benchmark.append({
                'content': '\n'.join(lines_in_window),
                'metadata': {
                    'namespace': task['namespace'],
                    'type': task['type'],  # 'function'
                    'repo_path': repo_path,
                    'relative_path': task['relative_path'],
                    'start_line_no': start_line_no,
                    'end_line_no': line_no,
                    'window_size': self.window_size,
                    'signature_position': task['signature_position'],
                    'body_position': task['body_position'],
                    'dependency': task['dependency'],
                    'requirement': task['requirement'],
                    'tests': task['tests'],
                    'indent': task['indent']
                }
            })
        print(f'Build {len(windows_in_benchmark)} benchmark windows for {os.path.basename(self.repo)} with window size {self.window_size}.')
        output_path = PathUtils.get_benchmark_window_path(self.repo, self.window_size)
        FileUtils.dump_pickle(windows_in_benchmark, output_path)

class GroundTruthWindowBuilder:
    def __init__(self, metadata, repo, window_size):
        self.tasks = metadata
        self.repo = repo
        self.window_size = window_size

    def build_windows(self):
        windows_in_benchmark = []
        for task in self.tasks:
            repo_path = os.path.join(Globals.REPO_BASE_DIR, task['project_path'])
            if self.repo != repo_path:
                continue
            file_path = os.path.join(Globals.REPO_BASE_DIR, task['completion_path'])
            code = FileUtils.read_file_as_string(file_path)
            code_lines = code.splitlines()
            start_line_no = task['signature_position'][0] - 1
            end_line_no = min(start_line_no + self.window_size, len(code_lines))
            lines_in_window = code_lines[start_line_no:end_line_no]  # do not consider code after the method
            windows_in_benchmark.append({
                'content': '\n'.join(lines_in_window),
                'metadata': {
                    'namespace': task['namespace'],
                    'type': task['type'],  # 'function'
                    'repo_path': repo_path,
                    'relative_path': task['completion_path'],
                    'start_line_no': start_line_no,
                    'end_line_no': end_line_no,
                    'window_size': self.window_size,
                    'signature_position': [task['signature_position'][0] - 1, task['signature_position'][1]],  # deveval starts from 1
                    'body_position': [task['body_position'][0] - 1, task['body_position'][1]],
                    'dependency': task['dependency'],
                    'requirement': task['requirement'],
                    'tests': task['tests'],
                    'indent': task['indent']
                }
            })
        print(f'Build {len(windows_in_benchmark)} benchmark windows for {os.path.basename(self.repo)} with window size {self.window_size}.')
        output_path = PathUtils.get_ground_truth_window_path(self.repo, self.window_size)
        FileUtils.dump_pickle(windows_in_benchmark, output_path)


class PredictionWindowBuilder:
    def __init__(self, prediction_path, repo, window_size, slice_size):
        self.repo = repo
        self.prediction_path = prediction_path
        self.window_size = window_size
        self.slice_size = slice_size
        self.slice_step = window_size // slice_size

    def build_windows(self, bench='deveval'):
        repo_base_dir = Globals.REPO_BASE_DIR if bench == 'deveval' else Globals.REPO_BASE_CODEREVAL_DIR
        windows = []
        predictions = FileUtils.read_jsonl(self.prediction_path)
        for prediction in predictions:
            repo_path = prediction['metadata']['repo_path']
            if self.repo != repo_path:
                continue
            namespace = prediction['metadata']['namespace']
            indent = prediction['metadata']['indent']
            func_name = namespace.split('.')[-1]

            file_path = os.path.join(repo_base_dir, prediction['metadata']['relative_path'])
            original_code = FileUtils.read_file_as_string(file_path)
            original_code_lines = original_code.splitlines()
            if bench == 'deveval':
                original_end_line_no = prediction['metadata']['body_position'][0]  # using the body position to locate the existing context
            else:
                original_end_line_no = prediction['metadata']['signature_position'][1]
            for pred in [sample['text'] for sample in prediction['predictions']]:
                func_code = DataUtils.extract_function_from_output(pred, func_name)
                completion = DataUtils.extract_completion(func_code, indent)
                pred_lines = [l for l in completion.splitlines() if l.strip()]
                code_lines = original_code_lines[:original_end_line_no] + pred_lines
                end_line_no = min(len(code_lines), original_end_line_no + self.slice_step)
                start_line_no = max(0, end_line_no - self.window_size)
                lines_in_window = code_lines[start_line_no:end_line_no]
                if not [l for l in lines_in_window]:  # all empty lines
                    continue
                windows.append({
                    'content': '\n'.join(lines_in_window),
                    'metadata': {
                        'namespace': prediction['metadata']['namespace'],
                        'type': prediction['metadata']['type'],
                        'repo_path': repo_path,
                        'relative_path': prediction['metadata']['relative_path'],
                        'start_line_no': start_line_no,   # update the line no of start line to the prediction result
                        'end_line_no': end_line_no,
                        'window_size': self.window_size,
                        'signature_position': prediction['metadata']['signature_position'],
                        'body_position': prediction['metadata']['body_position'],
                        'dependency': prediction['metadata']['dependency'],
                        'requirement': prediction['metadata']['requirement'],
                        'tests': prediction['metadata']['tests'],
                        'indent': prediction['metadata']['indent']
                    }
                })
                break  # build the window only use the first prediction
        print(f'Build {len(windows)} prediction windows for {os.path.basename(self.repo)} with window size {self.window_size}.')
        output_path = PathUtils.get_prediction_window_path(self.repo, self.prediction_path)
        FileUtils.dump_pickle(windows, output_path)


class CoTWindowBuilder:
    def __init__(self, repo):
        self.repo = repo

    def build_windows_from_existing(self, windows):
        repo_name = os.path.basename(self.repo)
        selected_windows = []
        for window in windows:
            repo_path = os.path.join(Globals.REPO_BASE_DIR, window['metadata']['repo_path'])
            if self.repo != repo_path:
                continue
            selected_windows.append(window)
        print(f'Build {len(selected_windows)} cot windows for {repo_name}.')
        output_path = os.path.join(Globals.COT_WINDOW_DIR, f"{repo_name}.pkl")
        FileUtils.dump_pickle(selected_windows, output_path)

class RequirementWindowBuilder:
    def __init__(self, metadata, repo, window_size):
        self.tasks = metadata
        self.repo = repo
        self.window_size = window_size

    def build_windows(self, bench='deveval'):
        repo_base_dir = Globals.REPO_BASE_DIR if bench == 'deveval' else Globals.REPO_BASE_CODEREVAL_DIR
        windows_in_benchmark = []
        for task in self.tasks:
            repo_path = os.path.join(repo_base_dir, task['repo_path'])
            if self.repo != repo_path:
                continue
            line_no = task['signature_position'][1]
            start_line_no = max(0, line_no - self.window_size)
            windows_in_benchmark.append({
                'content': task['requirement']['Functionality'] if bench == 'deveval' else task['requirement'],
                'metadata': {
                    'namespace': task['namespace'],
                    'type': task['type'],  # 'function'
                    'repo_path': repo_path,
                    'relative_path': task['relative_path'],
                    'start_line_no': start_line_no,
                    'end_line_no': line_no,
                    'window_size': self.window_size,
                    'signature_position': task['signature_position'],
                    'body_position': task['body_position'],
                    'dependency': task['dependency'],
                    'requirement': task['requirement'],
                    'tests': task['tests'],
                    'indent': task['indent']
                }
            })
        print(f'Build {len(windows_in_benchmark)} requirement windows for {os.path.basename(self.repo)}.')
        output_path = PathUtils.get_requirement_window_path(self.repo)
        FileUtils.dump_pickle(windows_in_benchmark, output_path)