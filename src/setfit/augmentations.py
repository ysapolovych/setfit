from collections import defaultdict
import numpy as np
import torch

def create_index_dict(lst):
    index_dict = defaultdict(list)
    for i, v in enumerate(lst):
        index_dict[v].append(i)
    return index_dict

def augment_embeddings(embeddings: np.ndarray, labels: list[str | list[int]],
                       max_combinations: int = -1, shuffle: bool = True):

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

        n_possible_pairs = int((label_embeddings.shape[0] * (label_embeddings.shape[0] - 1)) / 2)

        linear_combinations = np.zeros((n_possible_pairs, label_embeddings.shape[1]))

        k = 0
        for i in range(label_embeddings.shape[0]):
            for j in range(i+1, label_embeddings.shape[0]):
                coef = np.random.rand()
                lc = coef*label_embeddings[i] + (1-coef)*label_embeddings[j]
                linear_combinations[k, :] = lc
                k += 1

        if max_combinations < n_possible_pairs and max_combinations != -1:
            idx = np.random.randint(n_possible_pairs, size=max_combinations)
            linear_combinations = linear_combinations[idx, :]

        all_linear_combinations.append(linear_combinations)


        new_labels = [label_dict[label_str]] * linear_combinations.shape[0]
        all_new_labels.extend(new_labels)

    all_linear_combinations = np.concatenate(all_linear_combinations, axis=0)

    if shuffle:
        new_order = [i for i in range(len(all_new_labels))]
        np.random.shuffle(new_order)
        all_new_labels = [all_new_labels[i] for i in new_order]
        all_linear_combinations = all_linear_combinations[new_order, :]

    embeddings = np.concatenate([embeddings, all_linear_combinations], axis=0)
    labels.extend(all_new_labels)

    return embeddings, labels


def mixup(x, y, alpha, use_cuda=True):
    # https://github.com/facebookresearch/mixup-cifar10/blob/eaff31ab397a90fbc0a4aac71fb5311144b3608b/train.py#L119
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    num_examples = x.size()[0]

    if use_cuda:
        index = torch.randperm(num_examples).cuda()
    else:
        index = torch.randperm(num_examples)

    #mix = np.maximum(mix, np.ones_like(mix) - mix)
    x_aug = lam * x + (1 - lam) * x[index, :]
    y_aug = lam * y + (1 - lam) * y[index, :]
    return x_aug, y_aug


def mixup_np(x, y, alpha, output_tensor: bool = True, device: str = 'cuda:0'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    num_examples = x.shape[0]
    index = np.random.permutation(num_examples) #arange(0, num_examples)

    if isinstance(x, torch.Tensor):
        x = x.to('cpu').numpy()
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()

    # np.random.shuffle(index)

    x_aug = lam * x + (1 - lam) * x[index, :]
    y_aug = lam * y + (1 - lam) * y[index, :]

    if output_tensor:
        return torch.from_numpy(x_aug).to(device), torch.from_numpy(y_aug).to(device)

    return x_aug, y_aug


def rand_section(x_size, lam):
    cut_ratio = np.sqrt(1 - lam)

    r = int(np.round(x_size * cut_ratio))

    cx = np.random.randint(x_size)

    ind1 = np.clip(cx - r // 2, 0, x_size)
    ind2 = np.clip(cx + r // 2, 0, x_size)

    return ind1, ind2


def cutmix(x, y, alpha):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    rand_index = torch.randperm(len(x))
    target_a = y
    target_b = y[rand_index]

    ind1, ind2 = rand_section(x.shape[1], lam)

    x_aug = x.clone()

    x_aug[:, ind1:ind2] = x[rand_index, ind1:ind2]

    y_aug = target_a * lam + target_b * (1 - lam)

    return x_aug, y_aug


def rand_section_np(x_size, lam):
    cut_ratio = np.sqrt(1 - lam)

    r = int(np.round(x_size * cut_ratio))

    cx = np.random.randint(x_size)

    ind1 = np.clip(cx - r // 2, 0, x_size)
    ind2 = np.clip(cx + r // 2, 0, x_size)

    return ind1, ind2
#%%
def cutmix_np(x, y, alpha, output_tensor: bool = True, device: str = 'cuda:0'):
    """Used on batches

    :param x: _description_
    :param y: _description_
    :param alpha: _description_
    :return: _description_
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    if isinstance(x, torch.Tensor):
        x = x.to('cpu').numpy()
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()

    rand_index = np.random.permutation(x.shape[0])
    target_a = y
    target_b = y[rand_index]

    ind1, ind2 = rand_section_np(x.shape[1], lam)

    x_aug = x.copy()

    x_aug[:, ind1:ind2] = x[rand_index, ind1:ind2]

    y_aug = target_a * lam + target_b * (1 - lam)

    if output_tensor:
        return torch.from_numpy(x_aug).to(device), torch.from_numpy(y_aug).to(device)

    return x_aug, y_aug