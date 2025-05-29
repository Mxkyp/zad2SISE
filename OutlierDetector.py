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
        z_scores = np.abs(stats.zscore(data, axis=0, nan_policy='omit'))
        return np.any(z_scores > threshold, axis=1)

    def detect_outliers_iqr(self, data):
        """Detekcja outlierów metodą IQR"""
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = np.any((data < lower_bound) | (data > upper_bound), axis=1)
        return outliers

    def detect_outliers_isolation_forest(self, data):
        """Detekcja outlierów metodą Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(data) == -1
            return outliers
        except:
            print("Błąd Isolation Forest, używam Z-score")
            return self.detect_outliers_zscore(data)

    def detect_outliers_mahalanobis(self, data):
        """Detekcja outlierów metodą odległości Mahalanobisa"""
        try:
            mean = np.mean(data, axis=0)
            cov = np.cov(data.T)

            # Sprawdzenie czy macierz jest odwracalna
            if np.linalg.det(cov) == 0:
                print("Macierz kowariancji jest singularna, używam Z-score")
                return self.detect_outliers_zscore(data)

            inv_cov = np.linalg.inv(cov)
            diff = data - mean
            mahalanobis_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

            threshold = np.percentile(mahalanobis_dist, 95)
            return mahalanobis_dist > threshold
        except Exception as e:
            print(f"Błąd Mahalanobis: {e}, używam Z-score")
            return self.detect_outliers_zscore(data)

    def detect_outliers(self, X, Y):
        """Główna metoda detekcji outlierów"""
        combined_data = np.hstack([X, Y])

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

            outlier_votes = zscore_outliers.astype(int) + iqr_outliers.astype(int) + mahal_outliers.astype(int)
            outliers = outlier_votes >= 2

            print(f"Z-score wykrył: {np.sum(zscore_outliers)} outlierów")
            print(f"IQR wykrył: {np.sum(iqr_outliers)} outlierów")
            print(f"Mahalanobis wykrył: {np.sum(mahal_outliers)} outlierów")
            print(f"Kombinacja (≥2 metody): {np.sum(outliers)} outlierów")

        else:
            raise ValueError(f"Nieznana metoda detekcji outlierów: {self.method}")

        return outliers

    def outlierDetector(self, X, Y):
        """Metoda kompatybilna z kodem sieci neuronowej"""
        return self.filter_data(X, Y)

    def filter_data(self, X, Y):
        """Filtruje dane usuwając outliery"""
        print(f"Dane wejściowe: {len(X)} próbek")

        outliers = self.detect_outliers(X, Y)
        clean_indices = ~outliers

        outlier_percentage = np.sum(outliers) / len(X) * 100
        print(f"Wykryto {np.sum(outliers)} outlierów z {len(X)} próbek ({outlier_percentage:.2f}%)")

        X_clean = X[clean_indices]
        Y_clean = Y[clean_indices]

        print(f"Dane po filtracji: {len(X_clean)} próbek")

        return X_clean, Y_clean, clean_indices