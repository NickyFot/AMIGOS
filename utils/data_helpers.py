import numpy as np
import torch

np.random.seed(42)  # setting seed for consistency


def get_subject_idx(data: list) -> list:
    subjects = set()
    for video_segment in data:
        p_idx = video_segment['video_id'].split('_')[0]
        subjects.update([p_idx])
    return list(subjects)


def get_indices_in_set(data: list, subject_idx: list) -> list:
    indices = list()
    for i, datum in enumerate(data):
        p_idx = datum['video_id'].split('_')[0]
        if p_idx in subject_idx:
            indices.append(i)
    return indices


def rnd_train_test(data: list, ratio: float) -> tuple:
    subject_idx = get_subject_idx(data)
    subject_idx = np.asarray(subject_idx)
    np.random.shuffle(subject_idx)
    threshold = int(len(subject_idx)*ratio)
    train_subject, test_subject = subject_idx[:threshold], subject_idx[threshold:]

    assert len([value for value in train_subject if value in test_subject]) == 0

    train_idx = get_indices_in_set(data, train_subject)
    test_idx = get_indices_in_set(data, test_subject)
    return train_idx, test_idx
