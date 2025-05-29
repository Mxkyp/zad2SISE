import os.path
import glob as glob
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

class DataLoader:
    """Klasa do ładowania i przetwarzania danych"""

    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    @staticmethod
    def get_base_path():
        """Pobiera ścieżkę do folderu z danymi"""
        return os.path.join(os.path.dirname(__file__), 'dane')

    @staticmethod
    def load_data_excel():
        """Ładowanie danych z plików Excel"""
        base_path = DataLoader.get_base_path()

        print(f"Ścieżka bazowa: {os.path.abspath(base_path)}")

        # Ścieżki do folderów F8 i F10
        f8_path = os.path.join(base_path, "F8")
        f10_path = os.path.join(base_path, "F10")

        print(f"Poszukiwanie danych w:")
        print(f" - {os.path.abspath(f8_path)}")
        print(f" - {os.path.abspath(f10_path)}")

        def load_static_data():
            """Ładowanie danych statycznych (treningowych)"""
            all_data = []

            # Szukanie plików statycznych w F8 i F10
            for folder_path in [f8_path, f10_path]:
                if os.path.exists(folder_path):
                    static_files = glob.glob(os.path.join(folder_path, "*stat*.xlsx"))
                    print(f"\nZnaleziono {len(static_files)} plików statycznych w {folder_path}")

                    for file_path in static_files:
                        try:
                            df = pd.read_excel(file_path)
                            print(f"  Załadowano {os.path.basename(file_path)}: {len(df)} próbek")
                            all_data.append(df)
                        except Exception as e:
                            print(f"  Błąd ładowania {os.path.basename(file_path)}: {e}")
                else:
                    print(f"Folder {folder_path} nie istnieje")

            # Jeśli nie znaleziono plików w podfolderach F8/F10, szukaj bezpośrednio w folderze dane
            if not all_data:
                print(f"\nNie znaleziono plików w podfolderach, szukam bezpośrednio w {base_path}")
                if os.path.exists(base_path):
                    static_files = glob.glob(os.path.join(base_path, "*stat*.xlsx"))
                    print(f"Znaleziono {len(static_files)} plików statycznych w {base_path}")

                    for file_path in static_files:
                        try:
                            df = pd.read_excel(file_path)
                            print(f"  Załadowano {os.path.basename(file_path)}: {len(df)} próbek")
                            all_data.append(df)
                        except Exception as e:
                            print(f"  Błąd ładowania {os.path.basename(file_path)}: {e}")

            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                print(f"Łącznie próbek statycznych: {len(combined_data)}")
                return combined_data
            else:
                print("Brak danych statycznych!")
                return None

        def load_dynamic_data():
            """Ładowanie danych dynamicznych (testowych)"""
            all_data = []

            # Szukanie plików dynamicznych w F8 i F10
            for folder_path in [f8_path, f10_path]:
                if os.path.exists(folder_path):
                    # Pliki dynamiczne to te bez 'stat' i bez 'random' w nazwie
                    all_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
                    dynamic_files = [f for f in all_files
                                     if 'stat' not in os.path.basename(f) and 'random' not in os.path.basename(f)]

                    print(f"\nZnaleziono {len(dynamic_files)} plików dynamicznych w {folder_path}")

                    for file_path in dynamic_files:
                        try:
                            df = pd.read_excel(file_path)
                            print(f"  Załadowano {os.path.basename(file_path)}: {len(df)} próbek")
                            all_data.append(df)
                        except Exception as e:
                            print(f"  Błąd ładowania {os.path.basename(file_path)}: {e}")
                else:
                    print(f"Folder {folder_path} nie istnieje")

            # Jeśli nie znaleziono plików w podfolderach F8/F10, szukaj bezpośrednio w folderze dane
            if not all_data:
                print(f"\nNie znaleziono plików w podfolderach, szukam bezpośrednio w {base_path}")
                if os.path.exists(base_path):
                    all_files = glob.glob(os.path.join(base_path, "*.xlsx"))
                    dynamic_files = [f for f in all_files
                                     if 'stat' not in os.path.basename(f) and 'random' not in os.path.basename(f)]

                    print(f"Znaleziono {len(dynamic_files)} plików dynamicznych w {base_path}")

                    for file_path in dynamic_files:
                        try:
                            df = pd.read_excel(file_path)
                            print(f"  Załadowano {os.path.basename(file_path)}: {len(df)} próbek")
                            all_data.append(df)
                        except Exception as e:
                            print(f"  Błąd ładowania {os.path.basename(file_path)}: {e}")

            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                print(f"Łącznie próbek dynamicznych: {len(combined_data)}")
                return combined_data
            else:
                print("Brak danych dynamicznych!")
                return None

        static_df = load_static_data()
        dynamic_df = load_dynamic_data()

        return static_df, dynamic_df

    @staticmethod
    def prepare_data_excel(df):
        """Przygotowanie danych z formatu Excel"""
        if df is None or df.empty:
            print("Brak danych do przetworzenia!")
            return None, None, []

        # Definicja cech sensorycznych
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

        # Sprawdzenie dostępnych kolumn
        print(f"Dostępne kolumny w danych:")
        for col in df.columns:
            print(f"  - {col}")

        # Filtrowanie dostępnych cech
        available_features = [col for col in feature_cols if col in df.columns]
        print(f"\nDostępne cechy sensoryczne: {len(available_features)}/{len(feature_cols)}")

        if not available_features:
            print("Nie znaleziono żadnych cech sensorycznych!")
            # Fallback - użyj współrzędnych jako cech wejściowych
            if "data__coordinates__x" in df.columns and "data__coordinates__y" in df.columns:
                print("Używam współrzędnych jako cech wejściowych")
                available_features = ["data__coordinates__x", "data__coordinates__y"]
            else:
                return None, None, []

        # Przygotowanie cech X
        X = df[available_features].values
        X = np.nan_to_num(X, nan=0.0)  # Zastąpienie NaN zerami

        # Sprawdzenie czy istnieją kolumny referencyjne
        if "reference__x" not in df.columns or "reference__y" not in df.columns:
            print("Brak kolumn referencyjnych! Sprawdź strukturę danych.")
            return None, None, available_features

        if "data__coordinates__x" not in df.columns or "data__coordinates__y" not in df.columns:
            print("Brak kolumn współrzędnych! Sprawdź strukturę danych.")
            return None, None, available_features

        # Obliczenie błędów (różnica między zmierzonymi a rzeczywistymi współrzędnymi)
        df_clean = df.copy()
        df_clean["error_x"] = df_clean["data__coordinates__x"] - df_clean["reference__x"]
        df_clean["error_y"] = df_clean["data__coordinates__y"] - df_clean["reference__y"]
        Y = df_clean[["error_x", "error_y"]].values

        print(f"Kształt danych X: {X.shape}")
        print(f"Kształt danych Y: {Y.shape}")
        print(f"Średni błąd X: {np.mean(np.abs(Y[:, 0])):.4f}")
        print(f"Średni błąd Y: {np.mean(np.abs(Y[:, 1])):.4f}")

        return X, Y, available_features

    @staticmethod
    def prepare_training_testing_data():
        """Główna metoda do przygotowania danych treningowych i testowych"""
        print("=== ŁADOWANIE DANYCH ===")

        # Ładowanie danych Excel
        static_df, dynamic_df = DataLoader.load_data_excel()

        if static_df is None:
            raise Exception("Nie udało się załadować danych statycznych (treningowych)")
        if dynamic_df is None:
            raise Exception("Nie udało się załadować danych dynamicznych (testowych)")

        print("\n=== PRZETWARZANIE DANYCH TRENINGOWYCH ===")
        X_train, Y_train, features_train = DataLoader.prepare_data_excel(static_df)

        print("\n=== PRZETWARZANIE DANYCH TESTOWYCH ===")
        X_test, Y_test, features_test = DataLoader.prepare_data_excel(dynamic_df)

        if X_train is None or Y_train is None:
            raise Exception("Błąd przetwarzania danych treningowych")
        if X_test is None or Y_test is None:
            raise Exception("Błąd przetwarzania danych testowych")

        print(f"\n=== PODSUMOWANIE ===")
        print(f"Dane treningowe: {X_train.shape[0]} próbek, {X_train.shape[1]} cech")
        print(f"Dane testowe: {X_test.shape[0]} próbek, {X_test.shape[1]} cech")

        return X_train, Y_train, X_test, Y_test, features_train