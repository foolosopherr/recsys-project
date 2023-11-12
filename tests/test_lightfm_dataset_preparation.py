import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from lightfm.data import Dataset
from lightfm import LightFM
from sklearn.feature_extraction.text import CountVectorizer
import re

import pytest


from recsys_project.lightfm_dataset_preparation import *


def test_create_items_feature():
    data = {
        "genres": ["Action", "Comedy", "Drama"],
        "authors": ["Author 1", "Author 2", "Author 3"],
        "year": [2020, 2021, 2022],
    }
    test_df = pd.DataFrame(data)

    expected_result = pd.Series(
        ["Action,Author 1,2020", "Comedy,Author 2,2021", "Drama,Author 3,2022"]
    )

    result = test_df.apply(lambda row: create_items_feature(*row), axis=1)

    pd.testing.assert_series_equal(result, expected_result)


def test_create_users_feature():
    data = {
        "age": ["18-35", "age_unknown", "65+"],
        "sex": ["Мужской", "Женский", "sex_unknown"],
    }
    test_df = pd.DataFrame(data)

    expected_result = pd.Series(
        ["18-35,Мужской", "age_unknown,Женский", "65+,sex_unknown"]
    )

    result = test_df.apply(lambda row: create_users_feature(*row), axis=1)

    pd.testing.assert_series_equal(result, expected_result)


def test_custom_tokenizer():
    sample_text = "This is a sample text! It  includes special characters like @ and white spaces."

    expected_result = [
        "This",
        "is",
        "a",
        "sample",
        "text",
        "It",
        "includes",
        "special",
        "characters",
        "like",
        "and",
        "white",
        "spaces",
    ]

    result = custom_tokenizer(sample_text)

    assert result == expected_result
