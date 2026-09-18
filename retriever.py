from abc import ABC, abstractmethod
import copy
import numpy as np
import os
import torch

from extract_context import ContextExtractor
from myutils import FileUtils
from globals import Globals

class Retriever(ABC):
    def __init__(self, repo, query_lines, repo_lines, top_k, output_path):
        self.repo = repo
        self.query_lines = query_lines
        self.repo_lines = repo_lines
        self.top_k = top_k
        self.output_path = output_path

    @staticmethod
    @abstractmethod
    def get_embedding(data_line):
        pass

    @staticmethod
    @abstractmethod
    def get_similarity(obj1, obj2):
        pass

    @staticmethod
    def get_cosine_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2)

    @staticmethod
    @abstractmethod
    def is_available_context(query_line, repo_line):
        pass


class SparseRetriever(Retriever):
    def __init__(self, repo, query_lines, repo_lines, top_k, output_path):
        super().__init__(repo, query_lines, repo_lines, top_k, output_path)

    @staticmethod
    def get_similarity(list1, list2):
        """
        calculate jaccard similarity for two set of word vectors.
        """
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return float(intersection) / union


class DenseRetriever(Retriever):
    def __init__(self, repo, query_lines, repo_lines, top_k, output_path):
        super().__init__(repo, query_lines, repo_lines, top_k, output_path)

    @staticmethod
    def get_similarity(embedding1, embedding2):
        norm_embedding1 = torch.nn.functional.normalize(embedding1, p=2, dim=1)
        norm_embedding2 = torch.nn.functional.normalize(embedding2, p=2, dim=1)
        return torch.einsum("ac,bc->ab", norm_embedding1, norm_embedding2).item()


class RepoCoderRetriever(Retriever):
    def __init__(self, repo, query_lines, repo_lines, top_k, output_path):
        super().__init__(repo, query_lines, repo_lines, top_k, output_path)

    @staticmethod
    def is_available_context(query_line, repo_line):
        if isinstance(repo_line['metadata'], list):
            is_available = []
            for metadata in repo_line['metadata']:
                if metadata['relative_path'] != query_line['metadata']['relative_path']:
                    is_available.append(True)
                    continue
                if metadata['end_line_no'] < query_line['metadata']['signature_position'][0]:
                    is_available.append(True)
                else:
                    is_available.append(False)
            return all(is_available)
        else:
            if repo_line['metadata']['relative_path'] != query_line['metadata']['relative_path']:
                return True
            if repo_line['metadata']['end_line_no'] < query_line['metadata']['signature_position'][0]:
                return True
            else:
                return False

    def retrieve(self, add_retrieval_filter=False):
        print(f'Search repo {self.repo} with top {self.top_k} similar contents.')
        query_with_top_k_retrieved = []
        for query_line in self.query_lines:
            if add_retrieval_filter:
                repo_name = os.path.basename(self.repo)
                visitor_path = os.path.join(Globals.VISITOR_DIR, f'{repo_name}.pkl')
                visitor = FileUtils.load_pickle(visitor_path)
                extractor = ContextExtractor(query_line['metadata'], visitor)
                candidates = [c['metadata']['namespace'] for c in extractor.extract_retrieval_candidates()]
            else:
                candidates = []
                for line in self.repo_lines:
                    if not isinstance(line['metadata'], list):  # exclude repocoder
                        candidates.append(line['metadata']['namespace'])
                if len(candidates) > 0:
                    print(f"Search relevant contexts for {query_line['metadata']['namespace']} "
                          f"from {len(candidates)} items.")
            new_line = copy.deepcopy(query_line)
            query_embedding = self.get_embedding(query_line)
            top_k_retrieved = []
            for repo_line in self.repo_lines:
                if not isinstance(repo_line['metadata'], list):
                    if repo_line['metadata']['namespace'] not in candidates:
                        continue
                # skip unavailable context (only for non-filtered retrieval)
                if not self.is_available_context(query_line, repo_line):
                    continue
                repo_embedding = self.get_embedding(repo_line)
                similarity = self.get_similarity(query_embedding, repo_embedding)
                # top_k_retrieved.append((repo_line, similarity))  # not consistent with subsequent storage
                top_k_retrieved.append({
                    'content': repo_line['content'],
                    'metadata': repo_line['metadata'],
                    'similarity': similarity
                })
            top_k_retrieved = sorted(top_k_retrieved, key=lambda x: x['similarity'], reverse=True)[:self.top_k]
            new_line['top_k_retrieved'] = top_k_retrieved
            query_with_top_k_retrieved.append(new_line)
        FileUtils.dump_pickle(query_with_top_k_retrieved, self.output_path)


class RepoCoderSparseRetriever(SparseRetriever, RepoCoderRetriever):
    def __init__(self, repo, query_lines, repo_lines, top_k, output_path):
        super().__init__(repo, query_lines, repo_lines, top_k, output_path)

    @staticmethod
    def get_embedding(data_line):
        return np.array(data_line['data'][0]['embedding'])


class RepoCoderDenseRetriever(DenseRetriever, RepoCoderRetriever):
    def __init__(self, repo, query_lines, repo_lines, top_k, output_path):
        super().__init__(repo, query_lines, repo_lines, top_k, output_path)

    @staticmethod
    def get_embedding(data_line, by_step=False):
        if by_step:
            return data_line['data'][0]['embedding_by_step']
        else:
            return data_line['data'][0]['embedding']

    def retrieve_by_step(self, add_retrieval_filter=False):
        print(f'Search repo {self.repo} with top {self.top_k} similar contents.')
        query_with_top_k_retrieved = []
        for query_line in self.query_lines:
            if add_retrieval_filter:
                repo_name = os.path.basename(self.repo)
                visitor_path = os.path.join(Globals.VISITOR_DIR, f'{repo_name}.pkl')
                visitor = FileUtils.load_pickle(visitor_path)
                extractor = ContextExtractor(query_line['metadata'], visitor)
                # default only retrieve cross file ones
                candidates = [c['metadata']['namespace'] for c in extractor.extract_retrieval_candidates()]
            else:
                candidates = [line['metadata']['namespace'] for line in self.repo_lines]
                print(f"Search relevant contexts for {query_line['metadata']['namespace']} "
                      f"from {len(candidates)} items.")
            new_line = copy.deepcopy(query_line)
            query_embedding_by_step = self.get_embedding(query_line, by_step=True)
            # store retrieved results separately
            top_k_retrieved_by_step = []
            for query_embedding in query_embedding_by_step:
                top_k_retrieved = []
                for repo_line in self.repo_lines:
                    if repo_line['metadata']['namespace'] not in candidates:
                        continue
                    # skip unavailable context
                    if not self.is_available_context(query_line, repo_line):
                        continue
                    repo_embedding = self.get_embedding(repo_line)
                    similarity = self.get_similarity(query_embedding, repo_embedding)
                    top_k_retrieved.append({
                        'content': repo_line['content'],
                        'metadata': repo_line['metadata'],
                        'similarity': similarity
                    })
                top_k_retrieved = sorted(top_k_retrieved, key=lambda x: x['similarity'], reverse=True)[:self.top_k]
                top_k_retrieved_by_step.append(top_k_retrieved)
            new_line['top_k_retrieved_by_step'] = top_k_retrieved_by_step
            query_with_top_k_retrieved.append(new_line)
        FileUtils.dump_pickle(query_with_top_k_retrieved, self.output_path)