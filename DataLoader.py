import os.path
import glob as glob
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class DataLoader:
    """ Klasa do ładowania i przetwarzania danych """

    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    @staticmethod
    def load_data_csv():
        """ Ładowanie danych z plików CSV (zgodnie z my_functions.py) """
        columns = ["input_X", "input_Y", "expected_X", "expected_Y"]
        training_data = pd.DataFrame(columns=columns)
        testing_data = pd.DataFrame(columns=columns)

        static_folders_paths = ["./pomiary/f8/stat", "./pomiary/F10/stat"]
        dynamic_folders_paths = ["./dane/f8/dyn", "./dane/F10/dyn"]

        for i in range(2):
            static_folder_path = static_folders_paths[i]
            dynamic_folder_path = dynamic_folders_paths[i]

            if os.path.exists(static_folder_path):
                static_files = sorted(file for file in os.listdir(static_folder_path) if file.endswith(".csv"))
                for file in static_files:
                    file_path = os.path.join(static_folder_path, file)
                    temp_data_frame = pd.read_csv(file_path, names=columns)
                    training_data = pd.concat([training_data, temp_data_frame], ignore_index=True)

            if os.path.exists(dynamic_folder_path):
                dynamic_files = sorted(file for file in os.listdir(dynamic_folder_path) if file.endswith(".csv"))
                for file in dynamic_files:
                    file_path = os.path.join(dynamic_folder_path, file)
                    temp_data_frame = pd.read_csv(file_path, names=columns)
                    testing_data = pd.concat([testing_data, temp_data_frame], ignore_index=True)

        training_data = training_data.dropna()
        testing_data = testing_data.dropna()

        return training_data, testing_data

    @staticmethod
    def load_data_excel(folder_path="./pomiary/F8/"):
        """ Ładowanie dnych z plków Excel """

        def load_static_data():
            all_data = []
            static_files = glob.glob(f"{folder_path}*stat*.xlsx")
            print(f"Znaleziono {len(static_files)} plików statycznych")

            for file_path in static_files:
                try:
                    df = pd.read_excel(file_path, sheet_name="Sheet1")
                    print(f"Załadowano {file_path}: {len(df)} próbek")
                    all_data.append(df)
                except Exception as e:
                    print(f"Błąd ładowania {file_path}: {e}")

            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                print(f"Łączenie próbek statycznych: {len(combined_data)}")
                return combined_data
            return None

        def load_dynamic_data():
            all_data = []
            dynamic_files = [f for f in glob.glob(f"{folder_path}*.xlsx")
                             if 'stat' not in f and 'random' not in f]
            print(f"Znaleziono {len(dynamic_files)} plików dynamicznych")

            for file_path in dynamic_files:
                try:
                    df = pd.read_excel(file_path, sheet_name="Sheet1")
                    print(f"Załadowano {file_path}: {len(df)} próbek")
                    all_data.append(df)
                except Exception as e:
                    print(f"Błąd ładowania {file_path}: {e}")

            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                print(f"Łączenie próbek dynamicznych: {len(combined_data)}")
                return combined_data
            return None

        static_df = load_static_data()
        dynamic_df = load_dynamic_data()

        return static_df, dynamic_df

    @staticmethod
    def prepare_data_excel(df):
        """ Przygotowanie danych z formatu Excel """

        feature_cols = [
            "data__tagData__gyro__x",
            "data__tagData__gyro__y",
            "data__tagData__gyro__z",
            "data__tagData__magnetic__x",
            "data__tagData__magnetic__y",
            "data__tagData__magnetic__z",
            "data__tagData__quaternion__x",
            "data__tagData__quaternion__y",
            "data__tagData__quaternion__z",
            "data__tagData__quaternion__w",
            "data__tagData__linearAcceleration__x",
            "data__tagData__linearAcceleration__y",
            "data__tagData__linearAcceleration__z",
            "data__tagData__pressure",
        ]

        available_features = [col for col in feature_cols if col in df.columns]
        print(f"Dostępne cechy: {len(available_features)}/{len(feature_cols)}")

        X = df[available_features].values
        X = np.nan_to_num(X, nan=0.0)

        df["error_x"] = df["data__coordinates__x"] - df["reference__x"]
        df["error_y"] = df["data__coordinates__y"] - df["reference__y"]
        Y = df[["error_x", "error_y"]].values

        return X, Y, available_features

    @staticmethod
    def prepare_data_csv(training_data, testing_data):
        """ Przygotowanie danych z formatu CSV """
        X_train = training_data[["input_X", "input_Y"]].values
        Y_train = training_data[["expected_X", "expected_Y"]].values - training_data[["input_X", "input_Y"]].values

        X_test = testing_data[["input_X", "input_Y"]].values
        Y_test = testing_data[["expected_X", "expected_Y"]].values - testing_data[["input_X", "input_Y"]].values

        return X_train, Y_train, X_test, Y_test