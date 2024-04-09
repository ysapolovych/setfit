from collections import defaultdict
import numpy as np

def create_index_dict(lst):
    index_dict = defaultdict(list)
    for i, v in enumerate(lst):
        index_dict[v].append(i)
    return index_dict

def augment_embeddings(embeddings: np.ndarray, labels: list[str | list[int]],
                       max_combinations: int = -1):

    label_str_list = [str(t) for t in labels]
    label_dict = {str(t): t for t in labels}

    label_indices_dict = create_index_dict(label_str_list)

    all_new_labels = []
    all_linear_combinations = []

    for label_str, indices in label_indices_dict.items():

        label_embeddings = embeddings[indices, :]

        num_arrays = label_embeddings.shape[0]

        if num_arrays == 1:
            continue

        n_possible_pairs = (label_embeddings.shape[0] * (label_embeddings.shape[0] - 1)) / 2

        linear_combinations = np.zeros((n_possible_pairs, label_embeddings.shape[1]))

        k = 0
        for i in range(label_embeddings.shape[0]):
            k += 1
            for j in range(i+1, label_embeddings.shape[0]):
                coef = np.random.rand()
                lc = coef*label_embeddings[i] + (1-coef)*label_embeddings[j]
                linear_combinations[k, :] = lc
                k += 1

        if max_combinations < n_possible_pairs:
            idx = np.random.randint(n_possible_pairs, size=max_combinations)
            linear_combinations = linear_combinations[idx, :]

        all_linear_combinations.append(linear_combinations)


        new_labels = [label_dict[label_str]] * max_combinations
        all_new_labels.extend(new_labels)

    all_linear_combinations = np.concatenate(all_linear_combinations, axis=0)

    embeddings = np.concatenate([embeddings, all_linear_combinations], axis=0)
    labels.extend(all_new_labels)

    return embeddings, labels