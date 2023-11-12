from recsys_project.lightfm_model import random_search
import warnings
from scipy.sparse import csr_matrix, coo_matrix
from typing import Tuple
import types

warnings.filterwarnings("ignore")


def test_random_search():
    train = coo_matrix([[0, 1], [1, 0]])
    test = coo_matrix([[1, 0], [0, 1]])
    train_weights = coo_matrix([[0, 3], [2, 0]])
    users_features = csr_matrix([[1, 0], [0, 1]])
    items_features = csr_matrix([[1, 0], [0, 1]])
    num_samples = 3

    result_generator = random_search(
        train, test, train_weights, users_features, items_features, num_samples
    )

    assert isinstance(result_generator, types.GeneratorType)

    for result in result_generator:
        assert isinstance(result, Tuple)
