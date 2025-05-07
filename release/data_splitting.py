
import numpy as np
import pandas as pd
import networkx as nx
import random

import torch
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split

from typing import List, Dict, Union, Optional, Literal, Tuple

get_device = lambda: 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

class RandomSplitter:
    def __init__(
            self, 
            test_size: float=0.15, 
            dev_size: float=0.10,
            random_state: int=42,
            verbose: bool=True,
            **kwargs
        ):
        assert 0 < test_size < 1, 'test_size must be a float between 0 and 1'
        assert 0 < dev_size < 1, 'dev_size must be a float between 0 and 1'
        assert 0 < dev_size+test_size < 1, 'sum of test_size and dev_size must be less than 1'
        assert random_state > 0, 'random_state must be a positive integer'

        self.test_size = test_size
        self.dev_size = dev_size
        self.random_state = random_state
        self.verbose = verbose

    def split(self, 
        data: List[Dict[str, Union[str, List[Tuple[int]]]]],
        return_dict: bool=True
    ) -> Tuple[List[Dict[str, Union[str, List[Tuple[int]]]]]]:
        
        positive_examples = [doc for doc in data if sum(doc['labels']) > 0]
        negative_examples = [doc for doc in data if sum(doc['labels']) == 0]
        
        # split the positive examples
        n_ = len(positive_examples)
        n_test = int(self.test_size*n_)
        n_dev = int(self.dev_size*n_)

        train, temp = train_test_split(positive_examples, test_size=n_test+n_dev, random_state=self.random_state)
        dev, test = train_test_split(temp, test_size=n_test, random_state=self.random_state)
        del temp

        # split the negative examples
        n_ = len(negative_examples)
        n_test = int(self.test_size*n_)
        n_dev = int(self.dev_size*n_)

        train_negatives, temp_negatives = train_test_split(negative_examples, test_size=n_test+n_dev, random_state=self.random_state)
        dev_negatives, test_negatives = train_test_split(temp_negatives, test_size=n_test, random_state=self.random_state)
        del temp_negatives

        # add to positive examples
        train.extend(train_negatives)
        dev.extend(dev_negatives)
        test.extend(test_negatives)

        # resuffle (to mix positive and negative examples within splits)
        random.Random(self.random_state).shuffle(train)
        random.Random(self.random_state).shuffle(dev)
        random.Random(self.random_state).shuffle(test)

        if self.verbose:
            print('Train:', len(train)/len(data))
            print('Dev:', len(dev)/len(data))
            print('Test:', len(test)/len(data))

        if return_dict:
            return dict(train=train, dev=dev, test=test)
        
        return train, dev, test

class MinOverlapSplitter:
    def __init__(
            self, 
            test_size: float=0.15, 
            dev_size: float=0.10,
            similarity_threshold: float=0.80,
            max_component_size: int=50,
            embedding_model_name: str='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            device: Optional[Literal['cpu', 'cuda', 'mps']]=None,
            random_state: int=42,
            verbose: bool=True,
            **kwargs
        ):
        assert 0 < test_size < 1, 'test_size must be a float between 0 and 1'
        assert 0 < dev_size < 1, 'dev_size must be a float between 0 and 1'
        assert 0 < dev_size+test_size < 1, 'sum of test_size and dev_size must be less than 1'
        assert 0 < similarity_threshold < 1, 'similarity_threshold must be a float between 0 and 1'
        assert max_component_size > 0, 'max_component_size must be a positive integer'
        assert random_state > 0, 'random_state must be a positive integer'

        self.test_size = test_size
        self.dev_size = dev_size
        self.similarity_threshold = similarity_threshold
        self.max_component_size = max_component_size
        self.random_state = random_state
        self.verbose = verbose
        self.device = get_device if device is None else device
        self.model_name = embedding_model_name
        self.model = SentenceTransformer(embedding_model_name, model_kwargs={'device_map': 'auto'})

    def _embed(self, mentions: List[str]) -> torch.Tensor:
        return self.model.encode(mentions, convert_to_tensor=True, normalize_embeddings=True, batch_size=16, show_progress_bar=self.verbose)
    
    def _compute_similarities(self, mentions: List[str]) -> np.ndarray:
        embeddings = self._embed(mentions)
        similarities = torch.mm(embeddings, embeddings.T)
        return similarities.cpu().detach().numpy()
    
    def _get_components(
            self, 
            similarities: np.ndarray
        ) -> List[set]:
        
        # obtain initial connected components from similarities-based unique mention-level graph
        adjacency_matrix = np.where(similarities >= self.similarity_threshold, 1, 0)
        G = nx.from_numpy_array(adjacency_matrix)
        components = list(nx.connected_components(G))
        del G

        # create a dataframe that maps enitiy IDs to example IDs (i.e., sentence IDs)
        clustered_mentions = [
                # get indexes of `mentions` where element == `unique_mentions[c]`
            (i, [j for j, mention in self.mentions if mention == self.unique_mentions[c]])
            for i, comp in enumerate(components)
            for c in comp
        ]
        clustered_mentions = pd.DataFrame(clustered_mentions, columns=['entity_id', 'example_idx']).explode('example_idx')
        assert len(clustered_mentions) == len(self.mentions)

        # Create a graph where each node is a data_idx and edges connect notes based on shared entity IDs
        self.G_ = nx.Graph()
        for _, row in clustered_mentions.iterrows():
            # Ensure each data_idx appears as a node
            if row['example_idx'] not in self.G_:
                self.G_.add_node(row['example_idx'])
            # Connect nodes that share the same cluster_id
            same_cluster = clustered_mentions[clustered_mentions['entity_id'] == row['entity_id']]['example_idx'].tolist()
            for idx in set(same_cluster):
                if row['example_idx'] != idx:
                    self.G_.add_edge(row['example_idx'], idx)
        # Find connected components - these are the groups of data_idx that must be in the same split
        self.components = list(nx.connected_components(self.G_))
        return self.components
    
    def _spectral_partition(self, G: nx.Graph) -> Tuple[List[int], List[int]]:
        laplacian = nx.normalized_laplacian_matrix(G).astype(float)
        _, eigenvectors = np.linalg.eigh(laplacian.toarray())
        fiedler_vector = eigenvectors[:, 1]
        nodes = list(G.nodes())
        part1 = [nodes[i] for i in range(len(nodes)) if fiedler_vector[i] < 0]
        part2 = [nodes[i] for i in range(len(nodes)) if fiedler_vector[i] >= 0]
        return part1, part2
    
    def _partition_components(self, max_component_size: int=50, max_iter: int=100) -> List[set]:
        i = 0
        while any(len(comp) > self.max_component_size for comp in self.components):
            i += 1
            if i == max_iter:
                break
            largest_idx = max(range(len(self.components)), key=lambda i: len(self.components[i]))
            largest_component = self.components.pop(largest_idx)
            subgraph = self.G_.subgraph(largest_component)
            part1, part2 = self._spectral_partition(subgraph)
            
            # check that all elements in largest_component are either in part1 or in part2
            assert all(i in part1 or i in part2 for i in largest_component), 'Not all elements are in the partitions'

            self.components.extend([set(part1), set(part2)])
        
    def split(self, 
        data: List[Dict[str, Union[str, List[Tuple[int]]]]],
        return_dict: bool=True
    ) -> Tuple[List[Dict[str, Union[str, List[Tuple[int]]]]]]:
        
        positive_examples = [doc for doc in data if sum(doc['labels']) > 0]
        negative_examples = [doc for doc in data if sum(doc['labels']) == 0]

        self.mentions = [(i, ent) for i, doc in enumerate(positive_examples) for ent, typ in doc['entities']]
        self.unique_mentions = list(set(m[1] for m in self.mentions))

        # create connected self.components based on mention similarity threshold
        if self.verbose: print('computing mention similarities ...')
        similarities = self._compute_similarities(self.unique_mentions)

        if self.verbose: print('getting components...')
        self._get_components(similarities)

        largest_idx = max(range(len(self.components)), key=lambda i: len(self.components[i]))
        n_largest = len(self.components[largest_idx])
        if n_largest > self.max_component_size:
            if self.verbose: print('partitioning components...')
            self._partition_components()
        if self.verbose: print('obtained', len(self.components), 'components')
        del self.G_

        # reshuffle for randomness
        random.Random(self.random_state).shuffle(self.components)

        # split self.components
        component_sizes = [len(comp) for comp in self.components]
        weights = np.array(component_sizes)/sum(component_sizes)

        train_cutoff = 1-self.dev_size-self.test_size
        dev_cutoff = 1-self.test_size

        # separate the components into train, dev, and test sets
        i = np.where(np.cumsum(weights) > train_cutoff)[0][0]
        train_components = self.components[:i]
        i = np.where(np.cumsum(weights) > dev_cutoff)[0][0]
        dev_components = self.components[len(train_components):i]
        test_components = self.components[i:]

        # test that every component is in one of the splits
        assert len(train_components) + len(dev_components) + len(test_components) == len(self.components), 'Not all components are in the splits'

        # get the indexes of the examples in each split
        train_idx = list(set([idx for component in train_components for idx in component]))
        dev_idx = list(set([idx for component in dev_components for idx in component]))
        test_idx = list(set([idx for component in test_components for idx in component]))

        # get the examples in each split
        train = [positive_examples[i] for i in train_idx]
        dev = [positive_examples[i] for i in dev_idx]
        test = [positive_examples[i] for i in test_idx]

        # test that every positive example is in one of the splits
        assert len(train) + len(dev) + len(test) == len(positive_examples), 'Not all positive examples are in the splits'

        # train_sids = [doc['metadata']['sentence_id'] for doc in train]
        # dev_sids = [doc['metadata']['sentence_id'] for doc in dev]
        # test_sids = [doc['metadata']['sentence_id'] for doc in test]

        # assert len(set(train_sids).intersection(dev_sids)) == 0, 'Overlapping sentences between train and dev'
        # assert len(set(train_sids).intersection(test_sids)) == 0, 'Overlapping sentences between train and test'
        # assert len(set(dev_sids).intersection(test_sids)) == 0, 'Overlapping sentences between dev and test'

        # split the negative examples
        n_ = len(negative_examples)
        n_test = int(self.test_size*n_)
        n_dev = int(self.dev_size*n_)

        train_negatives, temp_negatives = train_test_split(negative_examples, test_size=n_test+n_dev, random_state=self.random_state)
        dev_negatives, test_negatives = train_test_split(temp_negatives, test_size=n_test, random_state=self.random_state)
        del temp_negatives

        # add to positive examples
        train.extend(train_negatives)
        dev.extend(dev_negatives)
        test.extend(test_negatives)

        # resuffle (to mix positive and negative examples within splits)
        random.Random(self.random_state).shuffle(train)
        random.Random(self.random_state).shuffle(dev)
        random.Random(self.random_state).shuffle(test)

        if self.verbose:
            print('Train:', len(train)/len(data))
            print('Dev:', len(dev)/len(data))
            print('Test:', len(test)/len(data))

        if return_dict:
            return dict(train=train, dev=dev, test=test)

        return train, dev, test

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Data splitting for token classification')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the data file')
    parser.add_argument('--splitter', type=str, choices=['random', 'minoverlap'], default='minoverlap', help='Type of splitter to use')
    parser.add_argument('--test_size', type=float, default=0.15, help='Proportion of data to use for testing')
    parser.add_argument('--dev_size', type=float, default=0.10, help='Proportion of data to use for validation')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--seed', type=int, default=42, help='Path to the output directory')

    args = parser.parse_args()
    
    # check arguments
    import os
    assert os.path.exists(args.input_file), 'Input file does not exist'

    assert 0 < args.test_size < 1, 'test_size must be in (0, 1)'
    assert 0 < args.dev_size < 1, 'dev_size must be in (0, 1)'
    assert args.test_size + args.dev_size < 1, 'sum of test_size and dev_size must be less than 1'

    from utils import read_jsonl, write_jsonl
    from soft_seqeval.utils.processing import extract_spans
    from schema import ID2LABEL as id2label

    def parse_record(d):
        return {
            'id': d['id'], 
            'tokens': d['tokens'], 
            'labels': d['labels']['BSCModel'],
            'metadata': {
                'sentence_id': d['metadata']['sentence_id']
            }
        }

    data = read_jsonl(args.input_file)

    data = [parse_record(d) for d in data]

    for doc in data:
        doc['encoded_labels'] = [id2label[l] for l in doc['labels']]
        ents = extract_spans(doc['encoded_labels'], doc['tokens'])
        doc['entities'] = [(' '.join(ent[0]), ent[1]) for ent in ents]

    if args.splitter == 'random':
        splitter = RandomSplitter(
            test_size=args.test_size, 
            dev_size=args.dev_size,
            random_state=args.seed,
            verbose=True
        )
    elif args.splitter == 'minoverlap':
        splitter = MinOverlapSplitter(
            test_size=args.test_size, 
            dev_size=args.dev_size,
            random_state=args.seed,
            verbose=True
        )
    else:
        raise ValueError('Unknown splitter type: {}'.format(args.splitter))
    
    # split the data
    splits = splitter.split(data, return_dict=True)

    
    os.makedirs(args.output_dir, exist_ok=True)
    for s, d in splits.items():
        write_jsonl(d, os.path.join('splits', s+'.jsonl'))
