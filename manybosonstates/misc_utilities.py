"""Miscellanea."""
import itertools
import numpy as np


def uneven_chunker(iterable, chunks_list):
    group_maker = iter(iterable)
    for chunk_size in chunks_list:
        yield itertools.islice(group_maker, int(chunk_size))


def truncate_elements(input_array, n_elements_to_keep):
    for idx in range(len(input_array)):
        input_array[idx] = input_array[idx][:n_elements_to_keep]
