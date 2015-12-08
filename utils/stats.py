import numpy as np


def compute_confusion_matrix(true_labels, predictions, n_classes):
    """
    Assume that classes are identified from 0 to n_classes - 1
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.float32)
    for true_label, prediction in zip(true_labels, predictions):
        cm[true_label, prediction] += 1
    return cm


def get_stats(true_labels, predictions, n_classes):
    cm = compute_confusion_matrix(true_labels, predictions, n_classes)
    return precision(cm), recall(cm), accuracy(cm), f_score(cm)


def precision(confusion_matrix):
    """
    tp / (tp + fp)
    """
    n_classes = confusion_matrix.shape[0]
    return np.array([confusion_matrix[i, i] / confusion_matrix[:, i].sum() for i in xrange(n_classes)])


def recall(confusion_matrix):
    """
    tp / (tp + fn)
    """
    n_classes = confusion_matrix.shape[0]
    return np.array([confusion_matrix[i, i] / confusion_matrix[i, :].sum() for i in xrange(n_classes)])


def accuracy(confusion_matrix):
    """
    (tp + tn) / (tp + tn + fp + fn)
    """
    n_classes = confusion_matrix.shape[0]
    return np.array([(confusion_matrix.sum() - confusion_matrix[:, i].sum() + confusion_matrix[i, i] * 2 -
                      confusion_matrix[i, :].sum()) / confusion_matrix.sum() for i in xrange(n_classes)])


def f_score(confusion_matrix):
    """
    2 * precision * recall / (precision + recall)
    """
    p_list = precision(confusion_matrix)
    r_list = recall(confusion_matrix)
    return np.array([2 * p * r / (p + r) for p, r in zip(p_list, r_list)])