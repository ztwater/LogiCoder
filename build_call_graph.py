import os
import time

from myutils import FileUtils, AnalUtils
from globals import Globals


class CallGraphBuilder:
    def __init__(self, bench='deveval'):
        self.bench = bench
        if self.bench == 'deveval':
            self.repo_base_dir = Globals.REPO_BASE_DIR
        else:
            self.repo_base_dir = Globals.REPO_BASE_CODEREVAL_DIR

    def build_call_graphs(self, repo_path):
        """
        @param repo_path: the relative path from repo_base, e.g., neo4j---neo4j-python-driver,
            Database/awesome-autodl.
        """
        repo_name = os.path.basename(repo_path)
        print(f'Look into {repo_name} for knowledge base construction.')
        visitor_path = os.path.join(Globals.VISITOR_DIR, f'{repo_name}.pkl')
        # build the call graph visitor for each repository using pyan
        try:
            visitor = FileUtils.load_pickle(visitor_path)
        except FileNotFoundError:
            print('Build visitors for the first time.')
            start_time = time.time()
            visitor = AnalUtils.get_call_graph_visitor(os.path.join(self.repo_base_dir, repo_path))
            FileUtils.dump_pickle(visitor, visitor_path)
            build_time = time.time() - start_time
            print(f'  [TIMING] Built call graph visitor in {build_time:.2f}s')

        # get knowledge units (functions) and all call graph nodes
        unit_path = os.path.join(Globals.UNIT_DIR, f'{repo_name}_all_units.json')
        func_path = os.path.join(Globals.UNIT_DIR, f'{repo_name}_funcs.json')
        try:
            units = FileUtils.read_json(unit_path)
            funcs = FileUtils.read_json(func_path)
        except FileNotFoundError:
            print('Load units for the first time.')
            start_time = time.time()
            funcs, units = self.get_units(visitor.nodes)
            FileUtils.dump_json(units, unit_path)
            FileUtils.dump_json(funcs, func_path)
            load_time = time.time() - start_time
            print(f'  [TIMING] Load units in {load_time:.2f}s')

        # build bi-direction call graphs
        call_graph_path = os.path.join(Globals.CALL_GRAPH_DIR, f'{repo_name}_call_graph.json')
        called_graph_path = os.path.join(Globals.CALL_GRAPH_DIR, f'{repo_name}_called_graph.json')
        try:
            call_graph = FileUtils.read_json(call_graph_path)
            called_graph = FileUtils.read_json(called_graph_path)
        except FileNotFoundError:
            print('Build all graphs for the first time.')
            start_time = time.time()
            call_graph, called_graph = self.build_call_graph(visitor, units)
            FileUtils.dump_json(call_graph, call_graph_path)
            FileUtils.dump_json(called_graph, called_graph_path)
            build_time = time.time() - start_time
            print(f'  [TIMING] Built call graph in {build_time:.2f}s')

    def get_units(self, nodes):
        """
        Return all defined nodes in the current repository, as well as function-level ones.
        """
        funcs = dict()
        units = dict()
        for name in nodes:
            for node in nodes[name]:
                if AnalUtils.is_concrete_node(node):
                    units[node.get_name()] = AnalUtils.get_node_info(node, bench=self.bench)
                    # only keep functions
                    node_type = node.flavor.value
                    if not AnalUtils.is_function(node_type):
                        continue
                    # ignore all files under ../tests/.. directories (these could not be knowledge)
                    # rel_path = os.path.relpath(node.filename, Globals.REPO_BASE_DIR)
                    # parts = rel_path.split(os.sep)[:-1]
                    # if 'tests' in parts:
                    #     continue
                    funcs[node.get_name()] = AnalUtils.get_node_info(node, bench=self.bench)
        return funcs, units

    def build_call_graph(self, visitor, units):
        """
        Build a bi-direction call graph for all nodes in current repository.
        :param visitor: CallGraphVisitor, the call graph visitor parsed by pyan.
        :return: Tuple[Dict, Dict], the bi-direction call graphs.
        """
        call_graph = dict()
        called_graph = dict()
        for caller_namespace in visitor.uses_edges:
            if caller_namespace not in units:
                continue
            caller_info = units[caller_namespace]
            caller_file = caller_info['relative_path']
            for callee_node in visitor.uses_edges[caller_namespace]:
                if AnalUtils.is_concrete_node(callee_node):  # and callee_node.namespace != "*"
                    callee_namespace = callee_node.get_name()
                    if callee_namespace not in units:
                        print(f"Add a callee node {callee_node.get_name()} to all units.")
                        callee_info = AnalUtils.get_node_info(callee_node, bench=self.bench)
                        # TODO: add it to the unit
                        units[callee_namespace] = callee_info
                    else:
                        callee_info = units[callee_namespace]
                    callee_file = callee_info['relative_path']
                    # if caller is the parent of the callee, it just calls its own member
                    if callee_file == caller_file and caller_namespace in callee_node.namespace:
                        pass
                    else:
                        if caller_namespace not in call_graph:
                            call_graph[caller_namespace] = {'use': [], 'virtual': dict(), 'import': []}
                        call_graph[caller_namespace]['use'].append(callee_namespace)
                        if callee_namespace not in called_graph:
                            called_graph[callee_namespace] = []
                        called_graph[callee_namespace].append(caller_namespace)

        # restore unclear callees from virtual edges, if the candidate is in the current repository, we add the caller
        # as its potential use.
        for caller_namespace in visitor.virtual_uses_edges:
            if caller_namespace not in units:
                continue
            caller_info = units[caller_namespace]
            caller_file = caller_info['relative_path']
            for unclear_callee_name, candidates in visitor.virtual_uses_edges[caller_namespace].items():
                for candidate in candidates:
                    # add non-virtual for called graph
                    if AnalUtils.is_concrete_node(candidate):
                        callee_namespace = candidate.get_name()
                        if callee_namespace not in units:
                            print(f"Add a callee node {candidate.get_name()} to all units.")
                            callee_info = AnalUtils.get_node_info(candidate, bench=self.bench)
                            # TODO: add it to the unit
                            units[callee_namespace] = callee_info
                        else:
                            callee_info = units[callee_namespace]
                        callee_file = callee_info['relative_path']
                        # if caller is the parent of the callee, it just calls its own member
                        if callee_file == caller_file and caller_namespace in candidate.namespace:
                            pass
                        else:
                            if caller_namespace not in call_graph:
                                call_graph[caller_namespace] = {'use': [], 'virtual': dict(), 'import': []}
                            call_graph[caller_namespace]['virtual'].setdefault(unclear_callee_name, [])
                            if callee_namespace not in call_graph[caller_namespace]['virtual'][unclear_callee_name]:
                                call_graph[caller_namespace]['virtual'][unclear_callee_name].append(callee_namespace)
                            if callee_namespace not in called_graph:
                                called_graph[callee_namespace] = []
                            called_graph[callee_namespace].append(caller_namespace)

        for caller_namespace in visitor.import_uses_edges:
            if caller_namespace not in units:
                continue
            for callee_name in visitor.import_uses_edges[caller_namespace]:
                callee_node = visitor.import_uses_edges[caller_namespace][callee_name]
                callee_namespace = callee_node.get_name()
                if callee_namespace not in units:
                    continue
                if caller_namespace not in call_graph:
                    call_graph[caller_namespace] = {'use': [], 'virtual': dict(), 'import': []}
                call_graph[caller_namespace]['import'].append(callee_namespace)
                if callee_namespace not in called_graph:
                    called_graph[callee_namespace] = []
                called_graph[callee_namespace].append(caller_namespace)
        return call_graph, called_graph


if __name__ == '__main__':
    bench = 'codereval'
    repos = FileUtils.get_repositories(bench)
    call_graph_builder = CallGraphBuilder(bench)
    for repo in repos:
        call_graph_builder.build_call_graphs(repo)
