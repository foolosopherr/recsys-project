import itertools
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k
from lightfm import LightFM
from typing import Tuple, Any, Dict, Generator
import numpy as np
import pickle
from scipy.sparse import csr_matrix, coo_matrix


def sample_hyperparameters() -> Generator:
    """
    Yield possible hyperparameter choices
    """

    while True:
        yield {
            "no_components": np.random.randint(16, 64),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": np.random.choice(["bpr", "warp"]),
            "learning_rate": np.random.exponential(0.05),
            "item_alpha": np.random.exponential(1e-8),
            "user_alpha": np.random.exponential(1e-8),
            "max_sampled": np.random.randint(5, 15),
            "num_epochs": np.random.randint(5, 20),
            "random_state": [42],
        }


def random_search(
    train: coo_matrix,
    test: coo_matrix,
    train_weights: coo_matrix,
    users_features: csr_matrix,
    items_features: csr_matrix,
    num_samples: int,
) -> Generator:
    """
    Hyperparam tuning
    """

    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        num_epochs = hyperparams.pop("num_epochs")

        model = LightFM(**hyperparams)
        model.fit(
            interactions=train,
            sample_weight=train_weights,
            item_features=items_features,
            user_features=users_features,
            verbose=True,
            epochs=num_epochs,
            num_threads=20,
        )

        auc_score_train = auc_score(
            model,
            train,
            user_features=users_features,
            item_features=items_features,
            train_interactions=train,
            num_threads=20,
            check_intersections=False,
        ).mean()

        auc_score_test = auc_score(
            model,
            test,
            user_features=users_features,
            item_features=items_features,
            train_interactions=train,
            num_threads=20,
            check_intersections=False,
        ).mean()

        train_precision = precision_at_k(
            model,
            train,
            user_features=users_features,
            item_features=items_features,
            k=10,
            num_threads=20,
            check_intersections=False,
        ).mean()
        test_precision = precision_at_k(
            model,
            test,
            user_features=users_features,
            item_features=items_features,
            k=10,
            num_threads=20,
            check_intersections=False,
        ).mean()

        hyperparams["num_epochs"] = num_epochs

        print(
            auc_score_train,
            auc_score_test,
            train_precision,
            test_precision,
            hyperparams,
            model,
        )

        yield (
            auc_score_train,
            auc_score_test,
            train_precision,
            test_precision,
            hyperparams,
            model,
        )


def lightfm_random_search(
    interactions: coo_matrix,
    weights: coo_matrix,
    users_features: csr_matrix,
    items_features: csr_matrix,
    test_percentage: float,
    num_samples: int,
) -> Tuple:
    """
    Getting the best hyperparams and model
    """

    train, test = random_train_test_split(
        interactions, test_percentage=test_percentage, random_state=42
    )
    train_weights, _ = random_train_test_split(
        weights, test_percentage=test_percentage, random_state=42
    )

    (
        auc_score_train,
        auc_score_test,
        train_precision,
        test_precision,
        hyperparams,
        model,
    ) = max(
        random_search(
            train, test, train_weights, users_features, items_features, num_samples
        ),
        key=lambda x: x[1],
    )

    print(f"Best model: Train AUC {auc_score_train:.5f}, Test AUC {auc_score_test:.5f}")
    print(
        f"            Train Precision@10 {train_precision:.5f}, Test Precision@10 {test_precision:.5f}"
    )

    return hyperparams, model


def get_lightfm_model(
    interactions: coo_matrix,
    weights: coo_matrix,
    users_features: csr_matrix,
    items_features: csr_matrix,
    hyperparams: Any,
    to_save: bool,
    path: str,
):
    """
    Getting final lightfm model
    """

    if hyperparams:
        num_epochs = hyperparams.pop("num_epochs")
        model = LightFM(**hyperparams)
    else:
        num_epochs = 15
        model = LightFM(random_state=42)

    model.fit(
        interactions=interactions,
        sample_weight=weights,
        item_features=items_features,
        user_features=users_features,
        verbose=True,
        epochs=num_epochs,
        num_threads=20,
    )

    if to_save:
        with open(path + "lightfm_model.pickle", "wb") as handle:
            pickle.dump(model, handle)

    return model
