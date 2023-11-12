import pandas as pd
import numpy as np
from lightfm.data import Dataset
from lightfm import LightFM
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Any, Tuple
from scipy.sparse import csr_matrix
import re


def create_items_feature(
    genres: str, authors: str, year: int
) -> str:
    """
    Feature concationation for further processing
    """

    concat = genres + "," + authors + "," + str(year)
    return concat


def create_users_feature(age: str, sex: str) -> str:
    """
    Feature concationation for further processing
    """

    concat = age + "," + sex
    return concat


def custom_tokenizer(text: str) -> List:
    """
    Custom tokenizer for Count Vectorizer from sklearn.
    Removing special characters and white spaces and splitting
    """

    text = re.sub(r"[^\w\s+]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.split()


def get_clean_datasets(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Importing datasets and creating new feature for items_df & users_df
    """

    interactions_df = pd.read_csv(path + "interactions_clean.csv")
    items_df = pd.read_csv(path + "items_clean.csv")
    users_df = pd.read_csv(path + "users_clean.csv")

    interactions_df = interactions_df[
        interactions_df["user_id"].isin(users_df["user_id"].unique())
    ]
    interactions_df = interactions_df[
        interactions_df["item_id"].isin(items_df["id"].unique())
    ]

    items_df["features"] = items_df.iloc[:, 2:].apply(
        lambda row: create_items_feature(*row), axis=1
    )
    users_df["features"] = users_df.iloc[:, 1:].apply(
        lambda row: create_users_feature(*row), axis=1
    )

    return interactions_df, items_df, users_df


def create_lightfm_dataset(
    path: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dataset,
    csr_matrix,
    csr_matrix,
    np.ndarray,
    np.ndarray,
    csr_matrix,
    csr_matrix,
]:
    """
    Creating lightfm dataset instance and getting coo matrices
    """

    interactions_df, items_df, users_df = get_clean_datasets(path=path)

    vectorizer_users = CountVectorizer(
        max_features=20000, lowercase=False, tokenizer=custom_tokenizer
    )
    vectorizer_items = CountVectorizer(
        max_features=50000, lowercase=False, tokenizer=custom_tokenizer
    )

    users_features = vectorizer_users.fit_transform(users_df["features"])
    items_features = vectorizer_items.fit_transform(items_df["features"])

    users_features_names = vectorizer_users.get_feature_names_out()
    items_features_names = vectorizer_items.get_feature_names_out()

    user_ids_buffered = np.array([x for x in interactions_df["user_id"].unique()])
    item_ids_buffered = np.array([x for x in interactions_df["item_id"].unique()])
    
    # user_ids_buffered = np.array(user_ids_buffered)
    # item_ids_buffered = np.array(item_ids_buffered)

    dataset = Dataset()

    dataset.fit(
        users=user_ids_buffered,
        items=item_ids_buffered,
        item_features=items_features_names,
        user_features=users_features_names,
    )

    (interactions, weights) = dataset.build_interactions(
        (
            (k, v, w)
            for k, v, w in zip(
                interactions_df["user_id"].values,
                interactions_df["item_id"].values,
                interactions_df["rating"].values + 1, # type: ignore[arg-type, operator]
            )
        )
    )

    return (
        interactions_df,
        items_df,
        users_df,
        dataset,
        users_features,
        items_features,
        user_ids_buffered,
        item_ids_buffered,
        interactions,
        weights,
    )
