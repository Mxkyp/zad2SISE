import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler


class OutlierDetector:
    """Klasa do detekcji i eliminacji błędnych pomiarów"""

    def __init__(self, method='combined', threshold=3.0):
        self.method = method
        self.threshold = threshold
        self.scaler = StandardScaler()

    def detect_outliers_zscore(self, data, threshold=3.0):
        """Detekcja outlierów metodą Z-score"""
        try:
            # Sprawdź czy dane zawierają NaN lub inf
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                print("Dane zawierają NaN lub inf - usuwam je przed obliczaniem Z-score")
                # Zamień NaN i inf na medianę kolumny
                data_clean = data.copy()
                for col in range(data.shape[1]):
                    col_data = data[:, col]
                    mask = np.isfinite(col_data)
                    if np.any(mask):
                        median_val = np.median(col_data[mask])
                        data_clean[~mask, col] = median_val
                    else:
                        data_clean[:, col] = 0.0
                data = data_clean

            z_scores = np.abs(stats.zscore(data, axis=0, nan_policy='omit'))

            # Sprawdź czy z_scores zawiera NaN
            if np.any(np.isnan(z_scores)):
                print("Z-scores zawierają NaN - zastępuję zerami")
                z_scores = np.nan_to_num(z_scores, nan=0.0)

            return np.any(z_scores > threshold, axis=1)
        except Exception as e:
            print(f"Błąd w Z-score: {e}")
            return np.zeros(len(data), dtype=bool)

    def detect_outliers_iqr(self, data):
        """Detekcja outlierów metodą IQR"""
        try:
            # Sprawdź czy dane zawierają NaN lub inf
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                print("Dane zawierają NaN lub inf - czyszczę je przed IQR")
                data_clean = data.copy()
                for col in range(data.shape[1]):
                    col_data = data[:, col]
                    mask = np.isfinite(col_data)
                    if np.any(mask):
                        median_val = np.median(col_data[mask])
                        data_clean[~mask, col] = median_val
                    else:
                        data_clean[:, col] = 0.0
                data = data_clean

            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1

            # Sprawdź czy IQR jest zbyt mały (może prowadzić do problemów)
            IQR = np.where(IQR < 1e-10, 1e-10, IQR)

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = np.any((data < lower_bound) | (data > upper_bound), axis=1)
            return outliers
        except Exception as e:
            print(f"Błąd w IQR: {e}")
            return np.zeros(len(data), dtype=bool)

    def detect_outliers_isolation_forest(self, data):
        """Detekcja outlierów metodą Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest

            # Sprawdź i wyczyść dane
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                print("Czyszczę dane przed Isolation Forest")
                data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                data_clean = data

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(data_clean) == -1
            return outliers
        except Exception as e:
            print(f"Błąd Isolation Forest: {e}, używam Z-score")
            return self.detect_outliers_zscore(data)

    def detect_outliers_mahalanobis(self, data):
        """Detekcja outlierów metodą odległości Mahalanobisa"""
        try:
            # Sprawdź i wyczyść dane
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                print("Czyszczę dane przed obliczaniem odległości Mahalanobisa")
                data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                data_clean = data.copy()

            # Sprawdź czy mamy wystarczająco danych
            if len(data_clean) < data_clean.shape[1] + 1:
                print("Za mało danych dla Mahalanobis, używam Z-score")
                return self.detect_outliers_zscore(data)

            mean = np.mean(data_clean, axis=0)

            # Sprawdź czy średnie są skończone
            if not np.all(np.isfinite(mean)):
                print("Średnie zawierają nieskończoności, używam Z-score")
                return self.detect_outliers_zscore(data)

            # Oblicz macierz kowariancji z regularyzacją
            cov = np.cov(data_clean.T)

            # Sprawdź czy macierz kowariancji jest prawidłowa
            if not np.all(np.isfinite(cov)):
                print("Macierz kowariancji zawiera nieskończoności, używam Z-score")
                return self.detect_outliers_zscore(data)

            # Dodaj regularyzację do macierzy kowariancji
            regularization = 1e-6 * np.eye(cov.shape[0])
            cov_reg = cov + regularization

            # Sprawdź wyznacznik
            det_cov = np.linalg.det(cov_reg)
            if np.abs(det_cov) < 1e-10:
                print("Macierz kowariancji jest prawie singularna, używam Z-score")
                return self.detect_outliers_zscore(data)

            try:
                inv_cov = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                print("Nie można odwrócić macierzy kowariancji, używam pseudo-odwrotności")
                try:
                    inv_cov = np.linalg.pinv(cov_reg)
                except:
                    print("Błąd pseudo-odwrotności, używam Z-score")
                    return self.detect_outliers_zscore(data)

            # Sprawdź czy odwrotność jest prawidłowa
            if not np.all(np.isfinite(inv_cov)):
                print("Odwrotność macierzy zawiera nieskończoności, używam Z-score")
                return self.detect_outliers_zscore(data)

            diff = data_clean - mean

            # Oblicz odległości Mahalanobisa
            try:
                mahalanobis_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
            except:
                print("Błąd obliczania odległości Mahalanobisa, używam Z-score")
                return self.detect_outliers_zscore(data)

            # Sprawdź czy odległości są prawidłowe
            if not np.all(np.isfinite(mahalanobis_dist)):
                print("Odległości Mahalanobisa zawierają nieskończoności, używam Z-score")
                return self.detect_outliers_zscore(data)

            # Ustaw próg na 95. percentyl
            if len(mahalanobis_dist) > 0:
                threshold = np.percentile(mahalanobis_dist, 95)
                if not np.isfinite(threshold):
                    print("Próg Mahalanobisa jest nieskończony, używam Z-score")
                    return self.detect_outliers_zscore(data)
                return mahalanobis_dist > threshold
            else:
                return np.zeros(len(data), dtype=bool)

        except Exception as e:
            print(f"Błąd Mahalanobis: {e}, używam Z-score")
            return self.detect_outliers_zscore(data)

    def detect_outliers(self, X, Y):
        """Główna metoda detekcji outlierów"""
        try:
            # Sprawdź czy dane wejściowe są prawidłowe
            if X is None or Y is None:
                print("Dane wejściowe są None")
                return np.zeros(len(X) if X is not None else 0, dtype=bool)

            if len(X) == 0 or len(Y) == 0:
                print("Dane wejściowe są puste")
                return np.zeros(0, dtype=bool)

            if len(X) != len(Y):
                print("Niezgodne długości X i Y")
                min_len = min(len(X), len(Y))
                X = X[:min_len]
                Y = Y[:min_len]

            combined_data = np.hstack([X, Y])

            # Sprawdź kształt danych
            if combined_data.shape[0] == 0:
                return np.zeros(0, dtype=bool)

            print(f"Dane do analizy outlierów: {combined_data.shape}")

            if self.method == 'zscore':
                outliers = self.detect_outliers_zscore(combined_data, self.threshold)
            elif self.method == 'iqr':
                outliers = self.detect_outliers_iqr(combined_data)
            elif self.method == 'isolation_forest':
                outliers = self.detect_outliers_isolation_forest(combined_data)
            elif self.method == 'mahalanobis':
                outliers = self.detect_outliers_mahalanobis(combined_data)
            elif self.method == 'combined':
                # Kombinacja metod — punkt outlierem, jeśli wykryty przez co najmniej 2 metody
                print("Stosowanie kombinacji metod detekcji outlierów...")

                zscore_outliers = self.detect_outliers_zscore(combined_data)
                iqr_outliers = self.detect_outliers_iqr(combined_data)
                mahal_outliers = self.detect_outliers_mahalanobis(combined_data)

                # Sprawdź czy wszystkie metody zwróciły prawidłowe wyniki
                if len(zscore_outliers) != len(combined_data):
                    zscore_outliers = np.zeros(len(combined_data), dtype=bool)
                if len(iqr_outliers) != len(combined_data):
                    iqr_outliers = np.zeros(len(combined_data), dtype=bool)
                if len(mahal_outliers) != len(combined_data):
                    mahal_outliers = np.zeros(len(combined_data), dtype=bool)

                outlier_votes = zscore_outliers.astype(int) + iqr_outliers.astype(int) + mahal_outliers.astype(int)
                outliers = outlier_votes >= 2

                print(f"Z-score wykrył: {np.sum(zscore_outliers)} outlierów")
                print(f"IQR wykrył: {np.sum(iqr_outliers)} outlierów")
                print(f"Mahalanobis wykrył: {np.sum(mahal_outliers)} outlierów")
                print(f"Kombinacja (≥2 metody): {np.sum(outliers)} outlierów")

            else:
                raise ValueError(f"Nieznana metoda detekcji outlierów: {self.method}")

            # Sprawdź czy wynik ma prawidłowy kształt
            if len(outliers) != len(combined_data):
                print(f"Błąd: nieprawidłowy kształt wyników outlierów ({len(outliers)} vs {len(combined_data)})")
                return np.zeros(len(combined_data), dtype=bool)

            return outliers

        except Exception as e:
            print(f"Błąd w detect_outliers: {e}")
            return np.zeros(len(X) if X is not None else 0, dtype=bool)

    def outlierDetector(self, X, Y):
        """Metoda kompatybilna z kodem sieci neuronowej"""
        return self.filter_data(X, Y)

    def filter_data(self, X, Y):
        """Filtruje dane usuwając outliery"""
        try:
            print(f"Dane wejściowe: {len(X)} próbek")

            outliers = self.detect_outliers(X, Y)
            clean_indices = ~outliers

            outlier_percentage = np.sum(outliers) / len(X) * 100 if len(X) > 0 else 0
            print(f"Wykryto {np.sum(outliers)} outlierów z {len(X)} próbek ({outlier_percentage:.2f}%)")

            X_clean = X[clean_indices]
            Y_clean = Y[clean_indices]

            print(f"Dane po filtracji: {len(X_clean)} próbek")

            return X_clean, Y_clean, clean_indices

        except Exception as e:
            print(f"Błąd w filter_data: {e}")
            print("Zwracam oryginalne dane bez filtracji")
            return X, Y, np.ones(len(X), dtype=bool)