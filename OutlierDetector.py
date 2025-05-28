from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import stats


class OutlierDetector:
    """ Klasa do detekcji i eliminacji błędnych pomiarów """

    def __init__(self, method='zscore', threshold=3.0):
        self.method = method
        self.threshold = threshold
        self.scaler = StandardScaler()

    def detect_outliers_zscore(self, data, threshold=3.0):
        """ Detekcja outlierów metodą Z-score """
        z_scores = np.abs(stats.zscore(data, axis=0, nan_policy='omit'))
        return np.any(z_scores > threshold, axis=1)

    def detect_outliers_iqr(self, data):
        """ Detekcja outlierów metodą IQR """
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = np.any((data < lower_bound) | (data > upper_bound), axis=1)
        return outliers

    def detect_outliers_isolation_forest(self, data):
        """ Detekcja outlierów metodą Isolation Forest """
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(data) == -1
        return outliers

    def detect_outliers_mahalanobis(self, data):
        """ Detekcja outlierów metodą odległości Mahalanobisa """
        try:
            mean = np.mean(data, axis=0)
            cov = np.cov(data.T)
            inv_cov = np.linalg.inv(cov)

            diff = data - mean
            mahalanobis_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

            threshold = np.percentile(mahalanobis_dist, 95)
            return mahalanobis_dist > threshold
        except:
            # Fallback do Z-score, jeśli macierz kowariancji jest singularna
            return self.detect_outliers_zscore(data)

    def detect_outliers(self, X, Y):
        """ Główna metoda detekcji outlierów """
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
            zscore_outliers = self.detect_outliers_zscore(combined_data)
            iqr_outliers = self.detect_outliers_iqr(combined_data)
            mahal_outliers = self.detect_outliers_mahalanobis(combined_data)

            outlier_votes = zscore_outliers.astype(int) + iqr_outliers.astype(int) + mahal_outliers.astype(int)
            outliers = outlier_votes >= 2
        else:
            raise ValueError("Nieznana metoda detekcji outlierów")

        return outliers

    def filte_data(self, X, Y):
        """ Filtruje dane usuwając outliery """
        outliers = self.detect_outliers(X, Y)
        clean_indices = ~outliers

        print(f"Wykryto {np.sum(outliers)} outlierów z {len(X)} próbek ({np.sum(outliers)/len(X)*100:.2f}%)")

        return X[clean_indices], Y[clean_indices], clean_indices


