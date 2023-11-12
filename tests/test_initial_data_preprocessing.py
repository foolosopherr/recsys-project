import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from simplemma import text_lemmatizer
import numpy as np
import nltk
import re
import pytest

nltk.download("stopwords")

from recsys_project.initial_data_preprocessing import *


@pytest.fixture
def test_csv_file():
    data = {
        "age": ["18_24", "25_34", "45_54", "65_inf"],
        "sex": [1.0, 0.0, None, 1.0],
    }
    df = pd.DataFrame(data)
    return df


def test_preprocess_users_dataset(test_csv_file):
    result = preprocess_users_dataset(test_csv_file)
    expected = pd.DataFrame(
        {
            "age": ["18-24", "25-34", "45-54", "65+"],
            "sex": ["Мужской", "Женский", "sex_unknown", "Мужской"],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_clean_text_without_lemmatization_and_stopwords():
    input_text = "This is a test sentence with special characters! It needs cleaning."
    expected_result = (
        "this is a test sentence with special characters it needs cleaning."
    )

    result = clean_text(
        input_text, to_lemmatize=False, to_title=False, remove_stopwords=False
    )
    assert result == expected_result


def test_clean_text_with_lemmatization_and_stopwords():
    input_text = "This is a test sentence with special characters! It needs cleaning."
    expected_result = "test sentence special character need clean ."

    result = clean_text(
        input_text, to_lemmatize=True, to_title=False, remove_stopwords=True
    )
    assert result == expected_result


def test_clean_text_with_title_case():
    input_text = "This is a test sentence with special characters! It needs cleaning."
    expected_result = (
        "This Is A Test Sentence With Special Characters It Needs Cleaning."
    )

    result = clean_text(
        input_text, to_lemmatize=False, to_title=True, remove_stopwords=False
    )
    assert result == expected_result


def test_clean_text_empty_input():
    input_text = ""
    expected_result = ""

    result = clean_text(
        input_text, to_lemmatize=False, to_title=False, remove_stopwords=False
    )
    assert result == expected_result


def test_clean_text_with_unicode_characters():
    input_text = "Привет, мир! This is a test sentence with ё and é."
    expected_result = "привет, мир this is a test sentence with ё and é."

    result = clean_text(
        input_text, to_lemmatize=False, to_title=False, remove_stopwords=False
    )
    assert result == expected_result


def test_fill_nan_genres():
    data = {
        "authors": ["Author 1", "Author 2", "Author 1", "Author 2", "Author 3"],
        "genres": ["Genre A", "Genre B", None, None, None],
    }
    items_df = pd.DataFrame(data)

    expected_data = {
        "authors": ["Author 1", "Author 2", "Author 1", "Author 2", "Author 3"],
        "genres": ["Genre A", "Genre B", "Genre A", "Genre B", "genres_unknown"],
    }
    expected_df = pd.DataFrame(expected_data)

    result = fill_nan_genres(items_df)

    pd.testing.assert_frame_equal(result, expected_df)


def test_fill_nan_genres_empty_dataframe():
    items_df = pd.DataFrame(columns=["authors", "genres"])

    expected_df = pd.DataFrame(columns=["authors", "genres"])

    result = fill_nan_genres(items_df)

    pd.testing.assert_frame_equal(result, expected_df)


def test_fill_nan_genres_no_nan_values():
    data = {
        "authors": ["Author 1", "Author 2", "Author 1", "Author 2", "Author 3"],
        "genres": ["Genre A", "Genre B", "Genre C", "Genre D", "Genre E"],
    }
    items_df = pd.DataFrame(data)

    result = fill_nan_genres(items_df)

    pd.testing.assert_frame_equal(result, items_df)


def test_change_genres_feature():
    data = {
        "genres": [
            "Гвэ) preparation,romance",
            "Mathematics,science",
            "History 8 класс",
            "fantasy,география,экономика",
        ],
    }
    items_df = pd.DataFrame(data)

    expected_data = {
        "genres": [
            "Подготока к экзаменам,Romance",
            "Mathematics,Science",
            "History",
            "Fantasy,География,Экономика",
        ],
    }
    expected_df = pd.DataFrame(expected_data)

    result = change_genres_feature(items_df)

    pd.testing.assert_frame_equal(result, expected_df)


def test_change_genres_feature_empty_dataframe():
    items_df = pd.DataFrame(columns=["genres"])

    expected_df = pd.DataFrame(columns=["genres"])

    result = change_genres_feature(items_df)

    pd.testing.assert_frame_equal(result, expected_df)


def test_change_genres_feature_already_normalized():
    data = {
        "genres": ["Romance,Science,History", "Fantasy,География,Экономика"],
    }
    items_df = pd.DataFrame(data)

    result = change_genres_feature(items_df)

    pd.testing.assert_frame_equal(result, items_df)


def test_preprocess_items_dataset_fill_authors():
    data = {
        "authors": [None, "Author 2", None, "Author 4"],
        "genres": ["Genre a", "Genre b", "Genre c", "Genre d"],
    }
    items_df = pd.DataFrame(data)

    expected_data = {
        "authors": ["Authors_Unknown", "Author 2", "Authors_Unknown", "Author 4"],
        "genres": ["Genre a", "Genre b", "Genre c", "Genre d"],
    }
    expected_df = pd.DataFrame(expected_data)

    result = preprocess_items_dataset(items_df)

    pd.testing.assert_frame_equal(result, expected_df)


def test_preprocess_items_dataset_clean_authors():
    data = {
        "authors": ["Author 1!", "Author 2?", "Author 3*", "Author 4"],
        "genres": ["Genre A", "Genre B", "Genre C", "Genre D"],
    }
    items_df = pd.DataFrame(data)

    expected_data = {
        "authors": ["Author 1", "Author 2", "Author 3", "Author 4"],
        "genres": ["Genre a", "Genre b", "Genre c", "Genre d"],
    }
    expected_df = pd.DataFrame(expected_data)

    result = preprocess_items_dataset(items_df)

    pd.testing.assert_frame_equal(result, expected_df)


def test_preprocess_interactions_dataset():
    data = {
        "user_id": [1, 1, 2, 2, 2],
        "item_id": [101, 102, 103, 104, 105],
        "progress": [10, 20, 30, 40, 50],
        "rating": [None, 3.0, None, None, 2.0],
        "start_date": [
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
        ],
    }
    interactions_df = pd.DataFrame(data)

    expected_data = {
        "user_id": [1, 1, 2, 2, 2],
        "item_id": [101, 102, 103, 104, 105],
        "rating": [0.0, 3.0, 0.0, 0.0, 2.0],
        "start_date": [
            pd.to_datetime("2023-01-01"),
            pd.to_datetime("2023-01-02"),
            pd.to_datetime("2023-01-03"),
            pd.to_datetime("2023-01-04"),
            pd.to_datetime("2023-01-05"),
        ],
        "year": [2023, 2023, 2023, 2023, 2023],
        "month": [1, 1, 1, 1, 1],
        "sequence": [1, 2, 1, 2, 3],
    }

    expected_df = pd.DataFrame(expected_data)

    result = preprocess_interactions_dataset(interactions_df)

    # Check if the result matches the expected DataFrame
    pd.testing.assert_frame_equal(result, expected_df)
