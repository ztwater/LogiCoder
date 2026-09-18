import ast
import glob
import os
import re

from globals import Globals
from myutils import AnalUtils, DataUtils, FileUtils
from pyan_zyf_v2.analyzer import CallGraphVisitor


class ContextExtractor:
    def __init__(self, metadata, visitor: CallGraphVisitor):
        self.metadata = metadata
        self.visitor = visitor
        self.repo_name = self.metadata['repo_path'].split('/')[-1]
        self.knowledge_base = FileUtils.read_jsonl(os.path.join(Globals.DOC_KNOWLEDGE_DIR, f'{self.repo_name}.jsonl'))

    def extract_in_file_skeleton(self, context_filter=None):
        """
        Extract an in-file context skeleton without any implementation details for planner.
        :return: str, the in-file completion context skeleton.
        """
        truncate_tuples = []
        for line in self.knowledge_base:
            if line['metadata']['relative_path'] == self.metadata['relative_path']:
                if (line['metadata']['start_line_no'] < self.metadata['body_position'][0] and
                        line['metadata']['namespace'] != self.metadata['namespace']):
                    # only truncate non-init methods and functions
                    if line['metadata']['type'] == 'class':
                        continue
                    if (line['metadata']['type'] == 'method' and
                            line['metadata']['namespace'].split('.')[-1] == '__init__'):
                        continue
                    if context_filter and line['metadata']['namespace'] in context_filter[self.metadata['namespace']]:
                        continue
                    pos = line['metadata']['start_line_no'], line['metadata']['end_line_no']
                    code = DataUtils.get_code_by_metadata(line['metadata'])
                    signature = ContextExtractor.extract_function_signature(code)
                    res, idx = self.check_nested_functions(truncate_tuples, pos)
                    if res == 'no overlap':
                        truncate_tuples.append((pos[0], pos[1], signature))
                    elif res == 'overwrite':
                        truncate_tuples.pop(idx)
                        truncate_tuples.append((pos[0], pos[1], signature))
        file_path = os.path.join(Globals.REPO_BASE_DIR, self.metadata['relative_path'])
        code = FileUtils.read_file_as_string(file_path)
        code_lines = code.splitlines()
        context_end_line_no = self.metadata['signature_position'][0]
        truncate_lines = code_lines[:context_end_line_no]
        truncate_tuples = list(sorted(truncate_tuples, key=lambda x: x[0], reverse=True))
        for t in truncate_tuples:
            truncate_lines = truncate_lines[:t[0]] + [t[2]] + truncate_lines[t[1]:]
        return '\n'.join(truncate_lines)

    @staticmethod
    def check_nested_functions(existed, pos):
        start_line_no, end_line_no = pos
        for i, t in enumerate(existed):
            if end_line_no <= t[0] or start_line_no >= t[1]:
                continue
            elif start_line_no >= t[0] and end_line_no <= t[1]:
                return 'keep original', -1
            elif start_line_no <= t[0] and end_line_no >= t[1]:
                return 'overwrite', i
            else:
                raise Exception("Wrong overlap")
        return 'no overlap', -1

    @staticmethod
    def extract_function_signature(code_str):
        """
        Extract the signature from a function.
        :param code_str: str, the code string.
        :return: str, a one-line function signature string.
        """
        pattern = r"def\s+\w+\s*\([^)]*\)[^:]*:"
        signature = re.search(pattern, code_str, re.MULTILINE)
        end_pos = signature.end() - 1
        return code_str[:end_pos]

    def extract_retrieval_candidates(self):
        imported_nodes = self.extract_imported_nodes()
        knowledge_list = self.extract_knowledge_by_node(imported_nodes)
        # knowledge_list = self.extract_imported_knowledge()
        knowledge_list.extend(self.extract_available_in_file_knowledge())
        knowledge_list = self.remove_duplicates(knowledge_list)
        return knowledge_list

    def extract_cross_file_retrieval_candidates(self):
        imported_nodes = self.extract_imported_nodes()
        knowledge_list = self.extract_knowledge_by_node(imported_nodes)
        knowledge_list = self.remove_duplicates(knowledge_list)
        return knowledge_list

    def extract_imported_knowledge(self):
        knowledge = []
        call_info_file = os.path.join(Globals.ROOT_DIR, 'Dependency_Data', self.metadata['repo_path'],
                                      'all_call_info.json')
        print(f"Visiting {os.path.basename(self.metadata['repo_path'])}.")
        call_info = FileUtils.read_json(call_info_file)
        file_call_info = call_info[os.path.relpath(self.metadata['relative_path'], self.metadata['repo_path'])]
        for node_name, node_metadata in file_call_info.items():
            if node_metadata['type'] == 'module':
                print(node_metadata['import'])
                for item in node_metadata['import']:
                    if item['type'] == 'module':
                        print(f"Add knowledge from module {item['path']}.")
                        knowledge.extend(self.extract_knowledge_by_file(os.path.join(self.metadata['repo_path'],
                                                                                     item['path'])))
                    elif AnalUtils.is_function(item['type']):
                        print(f"Add knowledge {item['name']}.")
                        knowledge.extend(self.extract_knowledge_by_namespace(item['name']))
        return knowledge

    def extract_imported_nodes(self):
        filename = os.path.join(Globals.REPO_BASE_DIR, self.metadata['relative_path'])
        # print(f"Visiting {filename}.")
        # iterate all nodes in the current repository to find current module
        nodes = self.visitor.nodes
        module = None
        for name in nodes:
            for node in nodes[name]:
                if node.defined and node.namespace is not None:
                    # only keep nodes from the current file
                    if node.flavor.value == 'module' and node.filename == filename:
                        # print(f"Found module: {node.get_name()}.")
                        module = node
                        break
            if module:
                break
        if not module:
            # print("Do not find corresponding module.")
            return []
            # raise Exception(f'Module {filename} not found.')

        imported_nodes = []
        for n in self.visitor.import_uses_edges:
            if n == module.get_name():
                # print(f"Found current module {n} on import graph.")
                for n2 in self.visitor.import_uses_edges[n]:
                    import_node = self.visitor.import_uses_edges[n][n2]
                    # print(import_node.defined, import_node.namespace)
                    if import_node.defined and import_node.namespace is not None:
                        imported_nodes.append(import_node)
        # print("Found nodes:", [n.get_name() for n in imported_nodes])
        return imported_nodes

    def extract_available_in_file_knowledge(self):
        available_knowledge = []
        for line in self.knowledge_base:
            if (line['metadata']['relative_path'] == self.metadata['relative_path'] and
                    line['metadata']['namespace'] != self.metadata['namespace']):
                if line['metadata']['namespace'].startswith('telethon.tl.types'):
                    continue
                available_knowledge.append(line)
        return available_knowledge

    def extract_unavailable_in_file_knowledge(self):
        unavailable_knowledge = []
        for line in self.knowledge_base:
            if line['metadata']['relative_path'] == self.metadata['relative_path']:
                if line['metadata']['start_line_no'] >= self.metadata['signature_position'][0]:
                    unavailable_knowledge.append(line)
        return unavailable_knowledge

    def extract_knowledge_by_file(self, filename):
        knowledge = []
        # print(f"Looking for all nodes in file {filename}.")
        for line in self.knowledge_base:
            if filename == line['metadata']['relative_path']:
                if line['metadata']['namespace'].startswith('telethon.tl.types'):
                    continue
                knowledge.append(line)
        if len(knowledge) == 0:
            pass
            # print(f'No knowledge found for file: {filename}.')
        return knowledge

    def extract_knowledge_by_namespace(self, namespace):
        knowledge = []
        for line in self.knowledge_base:
            if line['metadata']['namespace'].startswith(namespace):
                if line['metadata']['namespace'].startswith('telethon.tl.types'):
                    continue
                knowledge.append(line)
        if len(knowledge) == 0:
            pass
            # print(f'No knowledge found for namespace: {namespace}.')
        return knowledge

    def extract_knowledge_by_node(self, nodes):
        knowledge = []
        for n in nodes:
            if n.flavor.value == 'module':
                rel_path = os.path.relpath(n.filename, Globals.REPO_BASE_DIR)
                knowledge.extend(self.extract_knowledge_by_file(rel_path))
            elif n.flavor.value == 'class':
                knowledge.extend(self.extract_knowledge_by_namespace(n.get_name()))
            elif AnalUtils.is_function(n.flavor.value):
                knowledge.extend(self.extract_knowledge_by_namespace(n.get_name()))
            else:
                pass
                # print('Other types of dependencies: ', n.get_name())
        return knowledge

    @staticmethod
    def remove_duplicates(knowledge):
        k_dict = dict()
        for k in knowledge:
            k_dict[k['metadata']['namespace']] = k
        return k_dict.values()


if __name__ == '__main__':
    benchmark_metadata = FileUtils.read_jsonl(Globals.BENCHMARK_METADATA_PATH)
    updated_metadata = []
    for line in benchmark_metadata:
        updated_metadata.append(DataUtils.update_metadata(line))

    file_name = os.path.join(Globals.REPO_BASE_DIR, "Multimedia/Internet/boto/boto/regioninfo.py")
    repo_path = 'Internet/boto'
    repo_name = os.path.basename(repo_path)
    visitor_path = os.path.join(Globals.VISITOR_DIR, f'{repo_name}.pkl')
    visitor = FileUtils.load_pickle(visitor_path)
    for m in updated_metadata:
        if m['repo_path'] == repo_path and m['namespace'] == 'boto.elasticache.connect_to_region':
            extractor = ContextExtractor(m, visitor)
            imported_nodes = extractor.extract_imported_nodes()
            knowledge_list = extractor.extract_knowledge_by_node(imported_nodes)
            print(knowledge_list)
