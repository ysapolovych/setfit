from collections import defaultdict
import numpy as np

def create_index_dict(lst):
    index_dict = defaultdict(list)
    for i, v in enumerate(lst):
        index_dict[v].append(i)
    return index_dict

def augment_embeddings(embeddings: np.ndarray, labels: list[str | list[int]],
                       fraction: float = 0.5):
    label_str_list = [str(t) for t in labels]
    label_dict = {str(t):t for t in labels}

    label_indices_dict = create_index_dict(label_str_list)

    all_new_labels = []
    all_linear_combinations = []

    for label_str, indices in label_indices_dict.items():

        label_embeddings = embeddings[indices, :]

        num_arrays = label_embeddings.shape[0]

        if num_arrays == 1:
            continue

        num_combinations = int(round(num_arrays * fraction))

        # randomly select arrays for combinations
        selected_indices = np.random.choice(num_arrays, (num_combinations, 2), replace=False)

        # generate random coefficients
        coefficients = np.random.rand(num_combinations)
        print(coefficients)

        # normalize coefficients
        coefficients /= np.sum(coefficients)
        print(coefficients)

        # Perform linear combinations
        linear_combinations = np.zeros((num_combinations, label_embeddings.shape[1]))

        for i in range(num_combinations):
            idx1, idx2 = selected_indices[i]
            arr1, arr2 = label_embeddings[idx1], label_embeddings[idx2]
            lc = coefficients[i] * arr1 + (1 - coefficients[i]) * arr2
            linear_combinations[i, :] = lc
            # linear_combinations.append(linear_combination)

        all_linear_combinations.append(linear_combinations)

        new_labels = [label_dict[label_str]] * num_combinations
        all_new_labels.extend(new_labels)

    all_linear_combinations = np.concatenate(all_linear_combinations, axis=0)

    embeddings = np.concatenate([embeddings, all_linear_combinations], axis=0)
    labels = labels.extend(all_new_labels)

    return embeddings, labels