import os

from globals import Globals
from myutils import AnalUtils, DataUtils, FileUtils
from repository_cache import RepositoryDataCache


class FunctionExtractor:
    def __init__(self, metadata, shared_data=None, bench='deveval'):
        """
        Initialize FunctionExtractor.

        Args:
            metadata: Task metadata containing repo_path and other info
            shared_data: Optional pre-loaded repository data from RepositoryDataCache.
                        If None, data will be loaded from disk (backward compatible).
                        If provided, should contain: {visitor, funcs, call_graph}
        """
        self.metadata = metadata
        self.repo_name = self.metadata['repo_path'].split('/')[-1]
        self.bench = bench
        self.repo_base_dir = Globals.REPO_BASE_DIR if self.bench == 'deveval' else Globals.REPO_BASE_CODEREVAL_DIR

        if shared_data is None:
            # Backward compatible: load from disk
            visitor_path = os.path.join(Globals.VISITOR_DIR, f"{self.repo_name}.pkl")
            func_path = os.path.join(Globals.UNIT_DIR, f"{self.repo_name}_funcs.json")
            call_graph_path = os.path.join(Globals.CALL_GRAPH_DIR, f'{self.repo_name}_call_graph.json')
            self.visitor = FileUtils.load_pickle(visitor_path)
            self.funcs = FileUtils.read_json(func_path)
            self.call_graph = FileUtils.read_json(call_graph_path)
        else:
            # Use pre-loaded data from cache
            self.visitor = shared_data.get('visitor')
            self.funcs = shared_data.get('funcs')
            self.call_graph = shared_data.get('call_graph')

        # Build indexes for O(1) lookups
        self._build_function_indexes()

    def _build_function_indexes(self):
        """Build index structures for fast function lookups."""
        self.funcs_by_file = {}      # relative_path -> [func_metadata]
        self.funcs_by_class = {}     # class_name -> [func_metadata]
        self.funcs_by_namespace = {} # namespace -> func_metadata

        for func_ns, func_md in self.funcs.items():
            # Index by file
            rel_path = func_md['relative_path']
            if rel_path not in self.funcs_by_file:
                self.funcs_by_file[rel_path] = []
            self.funcs_by_file[rel_path].append(func_md)

            # Index by class
            if 'class' in func_md and func_md['class']:
                class_name = func_md['class']
                if class_name not in self.funcs_by_class:
                    self.funcs_by_class[class_name] = []
                self.funcs_by_class[class_name].append(func_md)

            # Index by namespace (1-to-1 mapping)
            self.funcs_by_namespace[func_ns] = func_md

    def extract_function_candidates(self):
        import time
        method_start = time.time()

        candidates = []

        # Step 1: Extract imported nodes
        step_start = time.time()
        imported_nodes = self.extract_imported_nodes()
        step_time = time.time() - step_start
        print(f"  [TIMING] extract_imported_nodes: {step_time*1000:.2f}ms ({len(imported_nodes)} nodes)")

        # Step 2: Extract functions by node
        step_start = time.time()
        imported_funcs = self.extract_functions_by_node(imported_nodes)
        candidates.extend(imported_funcs)
        step_time = time.time() - step_start
        print(f"  [TIMING] extract_functions_by_node: {step_time*1000:.2f}ms ({len(imported_funcs)} functions)")

        # Step 3: Extract associated functions
        step_start = time.time()
        associated_funcs = self.extract_associated_functions()
        candidates.extend(associated_funcs)
        step_time = time.time() - step_start
        print(f"  [TIMING] extract_associated_functions: {step_time*1000:.2f}ms ({len(associated_funcs)} functions)")

        # Step 4: Extract infile functions
        step_start = time.time()
        infile_funcs = self.extract_infile_functions()
        candidates.extend(infile_funcs)
        step_time = time.time() - step_start
        print(f"  [TIMING] extract_infile_functions: {step_time*1000:.2f}ms ({len(infile_funcs)} functions)")


        # Step 5: Remove duplicates
        step_start = time.time()
        candidates = self.remove_duplicates(candidates)
        step_time = time.time() - step_start
        print(f"  [TIMING] remove_duplicates: {step_time*1000:.2f}ms ({len(candidates)} unique functions)")

        total_time = time.time() - method_start
        print(f"  [TIMING] Total extract_function_candidates: {total_time*1000:.2f}ms")

        return candidates

    def extract_imported_nodes(self):
        filename = os.path.join(self.repo_base_dir, self.metadata['relative_path'])
        # print(f"Visiting {filename}.")
        # iterate all nodes in the current repository to find current module
        nodes = self.visitor.nodes
        module = set()
        for name in nodes:
            for node in nodes[name]:
                if node.defined and node.namespace is not None:
                    # only keep nodes from the current file
                    # if node.flavor.value == 'module' and node.filename == filename:
                    if node.filename == filename:
                        # print(f"Found module: {node.get_name()}.")
                        module.add(node)

        imported_nodes = set()
        # OPTIMIZED: Direct dictionary lookup instead of nested iteration
        for m in module:
            module_name = m.get_name()
            if module_name in self.visitor.import_uses_edges:  # O(1) lookup
                # print(f"Found current module {module_name} on import graph.")
                for n2 in self.visitor.import_uses_edges[module_name]:
                    import_node = self.visitor.import_uses_edges[module_name][n2]
                    # print(import_node.defined, import_node.namespace)
                    if import_node.defined and import_node.namespace is not None:
                        imported_nodes.add(import_node)
        # print("Found nodes:", [n.get_name() for n in imported_nodes])
        return imported_nodes

    def extract_associated_functions(self):
        funcs = []
        if self.metadata['type'] == 'function':
            return funcs
        else:
            current_class = '.'.join(self.metadata['namespace'].split(".")[:-1])
        funcs.extend(self.extract_associated_functions_for_single_item(current_class))
        for func_ns, func_md in self.funcs.items():
            if func_md['class'] == current_class:
                funcs.extend(self.extract_associated_functions_for_single_item(func_ns))
        return funcs

    def extract_infile_functions(self):
        """Extract functions from same file, excluding current function."""
        rel_path = self.metadata['relative_path']
        current_ns = self.metadata['namespace']

        # O(1) lookup using index
        all_funcs_in_file = self.funcs_by_file.get(rel_path, [])

        # Filter out current function
        return [f for f in all_funcs_in_file if f['namespace'] != current_ns]

    def extract_functions_by_node(self, nodes):
        funcs = []
        for n in nodes:
            if n.flavor.value == 'module':
                rel_path = os.path.relpath(n.filename, self.repo_base_dir)
                funcs.extend(self.extract_functions_by_file(rel_path))
            elif n.flavor.value == 'class':
                funcs.extend(self.extract_functions_by_class(n.get_name()))
            elif AnalUtils.is_function(n.flavor.value):
                funcs.extend(self.extract_functions_by_namespace(n.get_name()))
            else:
                pass
                # print('Other types of dependencies: ', n.get_name())
        return funcs

    def extract_functions_by_file(self, filename):
        """O(1) lookup using index."""
        return self.funcs_by_file.get(filename, []).copy()

    def extract_functions_by_class(self, namespace):
        """O(1) lookup using index."""
        return self.funcs_by_class.get(namespace, []).copy()

    def extract_functions_by_namespace(self, namespace):
        """O(1) lookup using index."""
        func_md = self.funcs_by_namespace.get(namespace)
        return [func_md] if func_md else []

    def extract_associated_functions_for_single_item(self, namespace):
        funcs = []
        if namespace not in self.call_graph:
            return funcs

        current_ns = self.metadata['namespace']
        seen = set()  # Track namespaces to avoid duplicates

        # Process 'use' dependencies
        for ns in self.call_graph[namespace]['use']:
            # Direct namespace lookup
            if ns in self.funcs_by_namespace and ns != current_ns:
                func_md = self.funcs_by_namespace[ns]
                if func_md['namespace'] not in seen:
                    funcs.append(func_md)
                    seen.add(func_md['namespace'])

            # Class lookup
            if ns in self.funcs_by_class:
                for func_md in self.funcs_by_class[ns]:
                    if func_md['namespace'] != current_ns and func_md['namespace'] not in seen:
                        funcs.append(func_md)
                        seen.add(func_md['namespace'])

        # Process 'import' dependencies
        for ns in self.call_graph[namespace]['import']:
            # Direct namespace lookup
            if ns in self.funcs_by_namespace and ns != current_ns:
                func_md = self.funcs_by_namespace[ns]
                if func_md['namespace'] not in seen:
                    funcs.append(func_md)
                    seen.add(func_md['namespace'])

            # Class lookup
            if ns in self.funcs_by_class:
                for func_md in self.funcs_by_class[ns]:
                    if func_md['namespace'] != current_ns and func_md['namespace'] not in seen:
                        funcs.append(func_md)
                        seen.add(func_md['namespace'])

        # Process 'virtual' dependencies
        for unclear_ns, v in self.call_graph[namespace]['virtual'].items():
            for ns in v:
                # Direct namespace lookup
                if ns in self.funcs_by_namespace and ns != current_ns:
                    func_md = self.funcs_by_namespace[ns]
                    if func_md['namespace'] not in seen:
                        funcs.append(func_md)
                        seen.add(func_md['namespace'])

                # Class lookup
                if ns in self.funcs_by_class:
                    for func_md in self.funcs_by_class[ns]:
                        if func_md['namespace'] != current_ns and func_md['namespace'] not in seen:
                            funcs.append(func_md)
                            seen.add(func_md['namespace'])

        return funcs

    @staticmethod
    def remove_duplicates(funcs):
        """Remove duplicates by namespace, filtering out telethon.tl.types."""
        func_dict = {}
        for f in funcs:
            ns = f['namespace']
            if not ns.startswith('telethon.tl.types'):
                func_dict[ns] = f
        return list(func_dict.values())


if __name__ == '__main__':
    metadata = FileUtils.read_jsonl(Globals.OUR_BENCHMARK_METADATA_PATH)
    start = 1
    for m in metadata[start:start+1]:
        print(m['namespace'])
        if m['namespace'] == 'boto.ec2.ec2object.TaggedEC2Object.add_tags':
            extractor = FunctionExtractor(m)
            candidates = extractor.extract_function_candidates()
            print([c['namespace'] for c in candidates])
        else:
            extractor = FunctionExtractor(m)
            candidates = extractor.extract_function_candidates()
            print([c['namespace'] for c in candidates])
