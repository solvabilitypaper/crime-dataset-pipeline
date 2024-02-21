# additional libraries
import json
import numpy as np
import pickle
import networkx as nx
from os.path import isfile, join
from os import listdir
import time
from tqdm import tqdm

# pykeen
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.sampling import BasicNegativeSampler
from pykeen.models import DistMult, ComplEx
import pykeen.utils as putils


def load_dataset(file_path: str,
                 has_headline: bool = False,
                 delimiter: str = ' ',
                 comments: str = '#',
                 keep_numerical: bool = False) -> np.array or None:
    # parse data to numpy array
    try:
        dataset = np.genfromtxt(file_path,
                                comments=comments,
                                delimiter=delimiter,
                                dtype=object if keep_numerical else str)
    except ValueError as e_message:
        raise ValueError(f'An error occurred while parsing the data: {e_message}')

    # remove potential headline
    if has_headline:
        dataset = dataset[1:]

    return dataset


def main():
    # init
    dataset_dir = '/data/crime-knowledge-graph/'
    prefix = 'crime_'
    start_over = True
    bar_format = '{l_bar}{bar:20}{r_bar}{bar:-10b}'

    # load existing files
    only_files = [file for file in listdir(dataset_dir) if isfile(join(dataset_dir, file))]
    identifiers = {file[:file.find('.')] for file in only_files if file.startswith(prefix)}
    identifiers = list(identifiers)
    identifiers.sort()

    # feature set
    feature_sets = {'feature01': ['has_category', 'has_in_50', 'nearby_100_restaurant']}

    for feature_set_id, feature_set in feature_sets.items():

        # remove already processed cases
        filtered_identifiers = []
        for identifier in identifiers:
            output_file = join(dataset_dir, f'{identifier}.graph.node_embeddings_{feature_set_id}.pkl')
            if not start_over and isfile(output_file):
                print(f'> Skip {identifier} because embeddings file already exists.')
                continue
            filtered_identifiers.append(identifier)

        for identifier in tqdm(filtered_identifiers,
                               total=len(filtered_identifiers),
                               desc='Experiments',
                               bar_format=bar_format):
            # Loads prepared training, testing and validation dataset.
            train_data = load_dataset(join(dataset_dir, f'{identifier}.graph.train.tsv'), delimiter='\t')
            valid_data = load_dataset(join(dataset_dir, f'{identifier}.graph.valid.tsv'), delimiter='\t')
            test_data = load_dataset(join(dataset_dir, f'{identifier}.graph.test.tsv'), delimiter='\t')

            # load relation to id dictionary
            with open(join(dataset_dir, f'{identifier}.graph.relation2id.txt')) as file:
                lines = [line.rstrip().split('\t') for line in file]
            relation_id = {relation: str(index) for relation, index in lines}
            feature_set_ids = [relation_id[entry] for entry in feature_set if entry in relation_id]

            # remove ground truth (and some other features)
            # we merge everything in a single array as we need to learn
            # embeddings for all nodes (in an unsupervised way)
            all_data = np.concatenate((train_data, valid_data, test_data), axis=0)
            row_idx = [idx
                       for idx, row in enumerate(all_data)
                       if row[1] not in feature_set_ids]
            all_data = np.delete(all_data, row_idx, axis=0)

            pykeen_triples = TriplesFactory.from_labeled_triples(all_data, create_inverse_triples=True)
            pykeen_training, pykeen_testing = TriplesFactory.split(pykeen_triples, ratios=0.999)

            # compute stats (on entire graph)
            edge_list = [(all_data[line][0], all_data[line][2]) for line in range(len(all_data))]
            graph = nx.MultiGraph()
            graph.add_nodes_from(all_data[:][0])
            graph.add_nodes_from(all_data[:][2])
            graph.add_edges_from(edge_list)  # default edge data=1
            num_nodes = graph.number_of_nodes()
            num_triples = graph.number_of_edges()

            # degree
            deg_list = list(dict(graph.degree(graph.nodes)).values())
            max_deg = np.max(deg_list)
            mean_deg = np.mean(deg_list)

            # degree centrality
            deg_clist = list(dict(nx.degree_centrality(graph)).values())
            mean_deg_c = np.mean(deg_clist)
            max_deg_c = np.max(deg_clist)
            min_deg_c = np.min(deg_clist)

            density = nx.density(graph)
            graph_stats = {'num_nodes': int(num_nodes),
                           'num_triples': int(num_triples),
                           'max_deg': int(max_deg),
                           'mean_deg': float(mean_deg),
                           'mean_deg_c': float(mean_deg_c),
                           'max_deg_c': float(max_deg_c),
                           'min_deg_c': float(min_deg_c),
                           'density': float(density)}
            with open(join(dataset_dir, f'{identifier}.graph_stats.json'), 'w', encoding='utf-8') as writer:
                writer.write(json.dumps(graph_stats, indent=4) + '\n')

            # randomly picked: randrange(2**32-1)
            # Note: The number of states (i.e., runs) has to match the number of
            # states/runs in step3
            random_states = [577090037, 1092444712010, 3639700191, 3445702192,
                             3280387012, 1102348763056, 782210080, 1704924261,
                             3059489832, 1112297082914]

            # learn embeddings
            node_indexer_all = dict()
            weights_all = dict()
            for random_state in random_states:
                # set seed
                putils.set_random_seed(random_state)

                # my model (DistMult, undirected)
                my_model = DistMult(
                    triples_factory=pykeen_training,
                    embedding_dim=100,
                    random_seed=random_state)

                neg_sampler = BasicNegativeSampler(
                    mapped_triples=pykeen_training.mapped_triples,
                    num_negs_per_pos=500)

                # init pipeline and train model
                result = pipeline(
                    training=pykeen_training,
                    testing=pykeen_testing,
                    model=my_model,
                    epochs=200,
                    negative_sampler=neg_sampler,
                    random_seed=random_state)

                # extract learned embeddings
                model = result.model
                entity_embedding_tensor = model.entity_representations[0](indices=None).detach().cpu().numpy()

                # save result
                weights_all[random_state] = entity_embedding_tensor
                node_indexer_all[random_state] = pykeen_training.entity_to_id

            with open(join(dataset_dir, f'{identifier}.graph.node_indexer_{feature_set_id}.pkl'), 'wb') as handle:
                pickle.dump(node_indexer_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(join(dataset_dir, f'{identifier}.graph.node_embeddings_{feature_set_id}.pkl'), 'wb') as handle:
                pickle.dump(weights_all, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f'Runtime: {time.time()-start}')
