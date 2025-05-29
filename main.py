import torch
import neuralNetwork as debilNetwork
import DataLoader as dl


def prepareTrainingData():
    df_training, df_testing = dl.load_all_data(dl.STATIC)

    df_training = df_training.dropna(
        subset=[
            "data__coordinates__x",
            "reference__x",
            "data__coordinates__y",
            "reference__y",
        ]
    )

    # --- Prepare training data ---
    inputs_train = torch.tensor(
        df_training[["data__coordinates__x", "data__coordinates__y"]].values,
        dtype=torch.float32,
    )
    targets_train = torch.tensor(
        df_training[["reference__x", "reference__y"]].values,
        dtype=torch.float32,
    )

    return inputs_train, targets_train


def main():
    inTrain, targetTrain, inTest, targetTest = prepareTrainingData()
    debilNetwork.train_model(inTrain, targetTrain, 70)


main()
