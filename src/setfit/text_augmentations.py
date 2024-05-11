import random
import string
import regex as re
from tokenize_uk import tokenize_text, tokenize_sents
from copy import deepcopy


def augment_text(x_train: list[str], y_train: list[str], x_add: list[str] = None, y_add: list[str] = None,
                 eda: bool = True, tfidf: bool = True, sent_swap: bool = True, tfidf_models: dict = None,
                 n: int = 2, max_word_swaps: int = 6, preserve_original: bool = True, **kwargs):
    x = deepcopy(x_train)
    y = deepcopy(y_train)

    if preserve_original:
        x_orig = x_train
        y_orig = y_train

    if x_add is not None and y_add is not None:
        x.extend(x_add)
        y.extend(y_add)


    if eda or sent_swap:
        x = [tokenize_sents(t) for t in x]

    x = x * n
    y = y * n

    if sent_swap:
        x = [sentence_shuffle(s) for s in x]
        if not eda:
            x = [' '.join(s) for s in x]

    if eda:
        processed_texts = []
        for old_sents in x:
            new_sents = []
            for s in old_sents:

                tokens = tokenize_text(s)[0][0]

                p = random.uniform(0, 0.6)
                n_changes = random.randint(0, max_word_swaps)
                if n_changes >= len(tokens):
                    n_changes = 1

                tokens = random_deletion(tokens, p)
                tokens = random_swap(tokens, n_changes)

                new_s = join_tokens(tokens)
                new_sents.append(new_s)
            processed_texts.append(' '.join(new_sents))
        x = processed_texts

    if tfidf:
        new_texts = []
        for t, lbl in zip(x, y):
            new_t = tf_idf_replacement(t, aug=tfidf_models[lbl])
            new_texts.append(new_t[0])

        x = new_texts

    if preserve_original:
        x = x_orig + x
        y = y_orig + y

    return x, y


def join_tokens(tokens):
    result = ""
    for i, token in enumerate(tokens):
        if token in string.punctuation and i != 0:
            result = result.rstrip()
        result += token + " "
    return result


def random_deletion(words, p):

    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        # Skip punctuation
        if not word.strip(string.punctuation):
            new_words.append(word)
            continue

        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words


def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words

    # Strip punctuation
    word1 = new_words[random_idx_1].strip(string.punctuation)
    word2 = new_words[random_idx_2].strip(string.punctuation)

    # Swap words
    new_words[random_idx_1] = word2 + new_words[random_idx_1][len(word1):]
    new_words[random_idx_2] = word1 + new_words[random_idx_2][len(word2):]

    return new_words


def sentence_shuffle(sents):
    new_sents = deepcopy(sents)
    random.shuffle(new_sents)
    return new_sents #' '.join(new_sents)


def tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)


def tf_idf_replacement(text, aug):
    return aug.augment(text)