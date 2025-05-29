import glob as glob
from sklearn.preprocessing import StandardScaler
import pandas as pd

BOTH = 0
STATIC = 1
DYNAMIC = 2


def load_all_data(flag):
    f8_path = "./pomiary/F8/"
    f10_path = "./pomiary/F10/"

    if flag == BOTH or flag == STATIC:
        # === Static Data ===
        static_files = glob.glob(f"{f8_path}*stat*.xlsx") + glob.glob(
            f"{f10_path}*stat*.xlsx"
        )
        print(f"Znaleziono {len(static_files)} plików statycznych")
        static_data = []
        for file_path in static_files:
            try:
                df = pd.read_excel(file_path, sheet_name="Sheet1")
                static_data.append(df)
            except Exception as e:
                print(f"Błąd ładowania {file_path}: {e}")
        static_combined = (
            pd.concat(static_data, ignore_index=True) if static_data else None
        )
        if static_combined is not None:
            print(f"Łączenie próbek statycznych: {len(static_combined)}")
        else:
            print("Nie znaleziono danych statycznych.")

    if flag == BOTH or flag == DYNAMIC:
        # === Dynamic Data ===
        dynamic_files = [
            f
            for f in glob.glob(f"{f8_path}*.xlsx") + glob.glob(f"{f10_path}*.xlsx")
            if "stat" not in f.lower() and "random" not in f.lower()
        ]
        print(f"Znaleziono {len(dynamic_files)} plików dynamicznych")
        dynamic_data = []
        for file_path in dynamic_files:
            try:
                df = pd.read_excel(file_path, sheet_name="Sheet1")
                dynamic_data.append(df)
            except Exception as e:
                print(f"Błąd ładowania {file_path}: {e}")
        dynamic_combined = (
            pd.concat(dynamic_data, ignore_index=True) if dynamic_data else None
        )
        if dynamic_combined is not None:
            print(f"Łączenie próbek dynamicznych: {len(dynamic_combined)}")
        else:
            print("Nie znaleziono danych dynamicznych.")

    if flag == STATIC:
        return static_combined
    if flag == DYNAMIC:
        return dynamic_combined
    if flag == BOTH:
        return static_combined, dynamic_combined
