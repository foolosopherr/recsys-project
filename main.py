from recsys_project.initial_data_preprocessing import preprocess_all_three_datasets
from recsys_project.lightfm_dataset_preparation import create_lightfm_dataset
from recsys_project.lightfm_model import get_lightfm_model, lightfm_random_search
import pickle


def main(path="data/", to_random_search=False, to_save=False):
    users_df, items_df, interactions_df = preprocess_all_three_datasets(
        path=path, to_save=False
    )
    print(users_df.shape, items_df.shape, interactions_df.shape)

    (
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
    ) = create_lightfm_dataset(path=path)

    if to_random_search:
        hyperparams, model = lightfm_random_search(
            interactions=interactions,
            weights=weights,
            users_features=users_features,
            items_features=items_features,
            test_percentage=0.1,
            num_samples=10,
        )
    else:
        model = get_lightfm_model(
            interactions=interactions,
            weights=weights,
            users_features=users_features,
            items_features=items_features,
            hyperparams=None,
            to_save=False,
            path=path,
        )

        hyperparams = model.get_params()

    if to_save:
        with open(path + "lightfm_model.pickle", "wb") as handle:
            pickle.dump(model, handle)

    print("Hyperparameters for LightFM model:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
