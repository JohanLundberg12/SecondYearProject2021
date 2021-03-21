from collections import Counter
import operator


def aggregate_counts_for_label(bags_of_words, y_train, label):
    """Aggregates counts for bags_of_words into a bag_of_words, given a label.
    Args:
        bags_of_words: list of counters
        y_train : list of labels.
        label : label to aggregate for.
    Returns:
        counter.
    """
    counts = Counter()
    for y, bow in zip(y_train, bags_of_words):
        if y == label:
            counts.update(bow)

    return counts


def bag_of_features(text, settings=None):
    """Count the number of feature occurences in the text. 
    Args: text (list): A list of tokens.
    Returns: a counter for the list. 
    """
    counter = Counter()
    if settings['unigram_freq']:
        for w in text:
            counter[w] += 1
    elif settings['unigram_pres']:
        for w in text:
            counter[w] = 1

    return counter

def filter_features(counts, n):
    """Filters out words from the the dictionary counts that 
        occur n or less times.
    Args:
        counts : A dictionary of counts of tokens.
        n (int) : A small number like 1-3. 
    Returns:
        dictionary : counts filtered.
    """
    filtered_counts = {k: v for k, v in counts.items() if v > n}
    sorted_counts = sorted(filtered_counts.items(), key=operator.itemgetter(1), reverse=True)
    counts_filtered = dict(sorted_counts)
    
    return counts_filtered

def aggregate_counts(bags_of_words):
    """Aggregates counts for bags_of_words into a single bag_of_words.
    Args:
        bags_of_words : A list of counters.
    Returns:
        counter.
    """
    counts = Counter()
    for bow in bags_of_words:
        counts.update(bow)

    counts = filter_features(counts, 3)
    features = []
    for bow in bags_of_words:
        temp = {}
        for word in bow:
            if word in counts:
                temp[word] = bow[word]
        features.append(temp)
    

    return features, counts

