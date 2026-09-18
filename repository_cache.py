"""
Repository-level data cache to avoid redundant loading of large data files.

This module provides a singleton cache that stores repository-specific data
(visitor, funcs, call_graph) to eliminate redundant disk I/O when processing
multiple tasks from the same repository.
"""

import os
from globals import Globals
from myutils import FileUtils


class RepositoryDataCache:
    """
    Singleton cache for repository-level data.

    Caches:
    - visitor: Parsed AST visitor data (pickle files, 0.3MB - 96MB)
    - funcs: Function metadata (JSON files)
    - call_graph: Call graph data (JSON files, up to 45MB)
    - called_graph: Called graph data (JSON files, up to 41MB)
    - mappings: Usage mappings (JSON files)

    Usage:
        cache = RepositoryDataCache.get_instance()
        data = cache.get_repository_data(repo_name)
        visitor = data['visitor']
        funcs = data['funcs']
    """

    _instance = None
    _cache = {}  # repo_name -> {visitor, funcs, call_graph, called_graph, mappings}

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of RepositoryDataCache."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def clear_cache(cls):
        """Clear all cached data to free memory."""
        cls._cache.clear()
        print(f"[RepositoryDataCache] Cache cleared")

    @classmethod
    def clear_repository(cls, repo_name):
        """Clear cached data for a specific repository."""
        if repo_name in cls._cache:
            del cls._cache[repo_name]
            print(f"[RepositoryDataCache] Cleared cache for {repo_name}")

    @classmethod
    def get_cache_stats(cls):
        """Get cache statistics."""
        return {
            'cached_repositories': len(cls._cache),
            'repository_names': list(cls._cache.keys())
        }

    def get_repository_data(self, repo_name, load_visitor=False, load_funcs=False,
                            load_call_graph=False, load_called_graph=False,
                            load_mappings=False):
        """
        Get repository data, loading from disk if not cached.

        Args:
            repo_name: Name of the repository
            load_visitor: Whether to load visitor pickle file
            load_funcs: Whether to load funcs JSON file
            load_call_graph: Whether to load call_graph JSON file
            load_called_graph: Whether to load called_graph JSON file
            load_mappings: Whether to load mappings JSON file

        Returns:
            Dictionary with requested data:
            {
                'visitor': visitor data (if load_visitor=True),
                'funcs': funcs data (if load_funcs=True),
                'call_graph': call_graph data (if load_call_graph=True),
                'called_graph': called_graph data (if load_called_graph=True),
                'mappings': mappings data (if load_mappings=True)
            }
        """
        # Check if repository is already cached
        if repo_name not in self._cache:
            print(f"[RepositoryDataCache] Loading data for repository: {repo_name}")
            self._cache[repo_name] = {}

            # Load visitor (pickle file)
            if load_visitor:
                visitor_path = os.path.join(Globals.VISITOR_DIR, f"{repo_name}.pkl")
                if os.path.exists(visitor_path):
                    self._cache[repo_name]['visitor'] = FileUtils.load_pickle(visitor_path)
                else:
                    self._cache[repo_name]['visitor'] = None
                    print(f"  [WARNING] Visitor file not found: {visitor_path}")

            # Load funcs (JSON file)
            if load_funcs:
                func_path = os.path.join(Globals.UNIT_DIR, f"{repo_name}_funcs.json")
                if os.path.exists(func_path):
                    self._cache[repo_name]['funcs'] = FileUtils.read_json(func_path)
                else:
                    self._cache[repo_name]['funcs'] = {}
                    print(f"  [WARNING] Funcs file not found: {func_path}")

            # Load call_graph (JSON file)
            if load_call_graph:
                call_graph_path = os.path.join(Globals.CALL_GRAPH_DIR, f'{repo_name}_call_graph.json')
                if os.path.exists(call_graph_path):
                    self._cache[repo_name]['call_graph'] = FileUtils.read_json(call_graph_path)
                else:
                    self._cache[repo_name]['call_graph'] = {}
                    print(f"  [WARNING] Call graph file not found: {call_graph_path}")

            # Load called_graph (JSON file)
            if load_called_graph:
                called_graph_path = os.path.join(Globals.CALL_GRAPH_DIR, f'{repo_name}_called_graph.json')
                if os.path.exists(called_graph_path):
                    self._cache[repo_name]['called_graph'] = FileUtils.read_json(called_graph_path)
                else:
                    self._cache[repo_name]['called_graph'] = {}
                    print(f"  [WARNING] Called graph file not found: {called_graph_path}")

            # Load mappings (JSON file)
            if load_mappings:
                mapping_path = os.path.join(Globals.MAPPING_DIR, f'{repo_name}.json')
                if os.path.exists(mapping_path):
                    self._cache[repo_name]['mappings'] = FileUtils.read_json(mapping_path)
                else:
                    self._cache[repo_name]['mappings'] = {}
                    print(f"  [WARNING] Mappings file not found: {mapping_path}")
        else:
            print(f"[RepositoryDataCache] Using cached data for repository: {repo_name}")

        return self._cache[repo_name]

    def get_visitor(self, repo_name):
        """Get visitor data for a repository."""
        data = self.get_repository_data(repo_name, load_visitor=True)
        return data.get('visitor')

    def get_funcs(self, repo_name):
        """Get funcs data for a repository."""
        data = self.get_repository_data(repo_name, load_funcs=True)
        return data.get('funcs')

    def get_call_graph(self, repo_name):
        """Get call_graph data for a repository."""
        data = self.get_repository_data(repo_name, load_call_graph=True)
        return data.get('call_graph')

    def get_called_graph(self, repo_name):
        """Get called_graph data for a repository."""
        data = self.get_repository_data(repo_name, load_called_graph=True)
        return data.get('called_graph')

    def get_mappings(self, repo_name):
        """Get mappings data for a repository."""
        data = self.get_repository_data(repo_name, load_mappings=True)
        return data.get('mappings')
