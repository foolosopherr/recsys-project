import pandas as pd
from simplemma import text_lemmatizer
import nltk
import re
from typing import Tuple, Any, Dict

nltk.download("stopwords")


def preprocess_users_dataset(users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing users dataset
    """

    age_replace = {
        "18_24": "18-24",
        "25_34": "25-34",
        "35_44": "35-44",
        "55_64": "55-64",
        "45_54": "45-54",
        "65_inf": "65+",
    }

    sex_replace = {1.0: "Мужской", 0.0: "Женский"}

    users_df["age"] = users_df["age"].replace(age_replace).fillna("age_unknown")
    users_df["sex"] = users_df["sex"].replace(sex_replace).fillna("sex_unknown")

    return users_df


def clean_text(
    text: str, to_lemmatize: bool, to_title: bool, remove_stopwords: bool
) -> str:
    """
    Cleaning string from special characters and white spaces
    There is an option to lemmatize and remove stopwords
    """

    text = re.sub(r"[^\w\s,.]", " ", text).lower()
    text = re.sub(r"\s+", " ", text).strip()

    if to_lemmatize:
        text = text_lemmatizer(text, lang=("ru", "en"))  # type: ignore[arg-type, assignment]
    else:
        text = text.split(" ")  # type: ignore[assignment]

    if remove_stopwords:
        stop_words_ru = set(nltk.corpus.stopwords.words("russian"))
        stop_words_en = set(nltk.corpus.stopwords.words("english"))

        stop_words = stop_words_ru.union(stop_words_en)
        text = [word for word in text if word not in stop_words]  # type: ignore[assignment]

    text = " ".join(text)
    if to_title:
        text = text.title()

    return text


def fill_nan_genres(items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill nan values in genres feature with most common genre for this author/authors
    """

    authors_with_nan_genres = items_df[items_df["genres"].isna()]["authors"]
    nan_genre_inds = authors_with_nan_genres.index
    authors_with_nan_genres = authors_with_nan_genres.values  # type: ignore[assignment]

    genres_author = {}

    for author in authors_with_nan_genres:
        genres_author_temp = (
            items_df[items_df["authors"] == author]["genres"].dropna().values
        )
        if len(genres_author_temp) == 0:
            to_replace = "genres_unknown"
        else:
            author_dct: Dict[Any, Any] = {}
            for genres in genres_author_temp:
                for genre in genres.split(","):
                    author_dct[genre] = author_dct.get(genre, 0) + 1

            to_replace = sorted(author_dct.items(), key=lambda x: x[1], reverse=True)[
                0
            ][0]

        genres_author[author] = to_replace

    items_df.loc[nan_genre_inds, "genres"] = items_df.loc[
        nan_genre_inds, "authors"
    ].apply(lambda x: genres_author[x])

    return items_df


def change_genres_feature(items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizing genres feature
    """

    for i in items_df.index:
        genres = items_df.loc[i, "genres"]
        temp_ = []
        for genre in genres.split(","):
            genre = genre.strip().lower()

            if "гвэ)" in genre or "егэ" in genre or "гиа" in genre:
                genre = "подготока к экзаменам"

            if " класс" in genre and "классика" not in genre:
                genre = genre.split()[:-2]
                genre = " ".join(genre)

            temp_.append(genre.capitalize())

        temp_ = ",".join(temp_)  # type: ignore[assignment]
        items_df.loc[i, "genres"] = temp_

    return items_df


def preprocess_items_dataset(items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing items dataset
    """

    items_df["authors"].fillna("authors_unknown", inplace=True)

    items_df["authors"] = items_df["authors"].apply(
        clean_text, to_lemmatize=False, to_title=True, remove_stopwords=False
    )

    genres_notna_inds = items_df[~items_df["genres"].isna()].index

    items_df.loc[genres_notna_inds, "genres"] = items_df.loc[
        genres_notna_inds, "genres"
    ].apply(clean_text, to_lemmatize=False, to_title=False, remove_stopwords=False)

    items_df = fill_nan_genres(items_df)
    items_df = change_genres_feature(items_df)

    return items_df


def preprocess_interactions_dataset(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing interactions dataset
    """

    interactions_df.fillna(0, inplace=True)

    interactions_df["start_date"] = pd.to_datetime(interactions_df["start_date"])
    interactions_df["year"] = interactions_df["start_date"].dt.year
    interactions_df["month"] = interactions_df["start_date"].dt.month

    interactions_df["sequence"] = interactions_df.groupby("user_id").cumcount() + 1

    interactions_df.drop(columns=["progress"], inplace=True)
    return interactions_df


def preprocess_all_three_datasets(
    path: str, to_save: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Importing and preparing all three datasets
    """
    
    try:
        users_df = pd.read_csv(path + "users.csv")
        print("Users CSV file loaded successfully")
    except FileNotFoundError:
        print("Users file not found. Please check the file path")
    except Exception as e:
        print(f"An error occurred while reading the Users CSV file: {e}")
        
    try:
        items_df = pd.read_csv(path + "items.csv")
        print("Items CSV file loaded successfully")
    except FileNotFoundError:
        print("Items file not found. Please check the file path")
    except Exception as e:
        print(f"An error occurred while reading the Items CSV file: {e}")
        
    try:
        interactions_df = pd.read_csv(path + "interactions.csv")
        print("Interactions CSV file loaded successfully")
    except FileNotFoundError:
        print("Interactions file not found. Please check the file path")
    except Exception as e:
        print(f"An error occurred while reading the Interactions CSV file: {e}")

    interactions_df = preprocess_interactions_dataset(interactions_df)
    users_df = preprocess_users_dataset(users_df)
    items_df = preprocess_items_dataset(items_df)

    if to_save:
        interactions_df.to_csv(path + "interactions_clean.csv", index=False)
        users_df.to_csv(path + "users_clean.csv", index=False)
        items_df.to_csv(path + "items_clean.csv", index=False)

    return users_df, items_df, interactions_df
