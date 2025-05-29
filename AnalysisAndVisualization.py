import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AnalysisAndVisualization:
    """Klasa do analizy i wizualizacji wyników"""

    @staticmethod
    def calculate_cdf(errors):
        """Obliczanie dystrybuanty (CDF)"""
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        return sorted_errors, cdf

    @staticmethod
    def calculate_error_metrics(Y_true, Y_pred, original_errors):
        """Obliczanie metryk błędów"""
        correction_errors = Y_true - Y_pred
        corrected_magnitude = np.sqrt(correction_errors[:, 0] ** 2 + correction_errors[:, 1] ** 2)
        original_magnitude = np.sqrt(original_errors[:, 0] ** 2 + original_errors[:, 1] ** 2)

        metrics = {
            'mse_original': np.mean(original_magnitude ** 2),
            'mse_corrected': np.mean(corrected_magnitude ** 2),
            'mae_original': np.mean(original_magnitude),
            'mae_corrected': np.mean(corrected_magnitude),
            'rmse_original': np.sqrt(np.mean(original_magnitude ** 2)),
            'rmse_corrected': np.sqrt(np.mean(corrected_magnitude ** 2)),
            'median_original': np.median(original_magnitude),
            'median_corrected': np.median(corrected_magnitude),
            'p90_original': np.percentile(original_magnitude, 90),
            'p90_corrected': np.percentile(corrected_magnitude, 90),
            'p95_original': np.percentile(original_magnitude, 95),
            'p95_corrected': np.percentile(corrected_magnitude, 95)
        }

        improvement = (metrics['mae_original'] - metrics['mae_corrected']) / metrics['mae_original'] * 100
        metrics['improvement_percent'] = improvement

        return metrics, original_magnitude, corrected_magnitude

    @staticmethod
    def plot_training_history(histories, model_names):
        """Wykres historii trenowania"""
        plt.figure(figsize=(15, 5))

        # Loss
        plt.subplot(1, 3, 1)
        for i, history in enumerate(histories):
            plt.plot(history.history['loss'], label=f'{model_names[i]} - trening')
            plt.plot(history.history['val_loss'], label=f'{model_names[i]} - walidacja', linestyle='--')
        plt.xlabel('Epoka')
        plt.ylabel('Loss (MSE)')
        plt.title('Historia trenowania - Loss')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)

        # MAE
        plt.subplot(1, 3, 2)
        for i, history in enumerate(histories):
            plt.plot(history.history['mae'], label=f'{model_names[i]} - trening')
            plt.plot(history.history['val_mae'], label=f'{model_names[i]} - walidacja', linestyle='--')
        plt.xlabel('Epoka')
        plt.ylabel('MAE')
        plt.title('Historia trenowania - MAE')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)

        # Learning Rate (jeśli dostępne)
        plt.subplot(1, 3, 3)
        for i, history in enumerate(histories):
            if 'lr' in history.history:
                plt.plot(history.history['lr'], label=f'{model_names[i]}')
        plt.xlabel('Epoka')
        plt.ylabel('Learning Rate')
        plt.title('Zmiana Learning Rate')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_error_comparison(models, X_test, Y_test, model_names):
        """Porównanie błędów dla różnych modeli"""
        plt.figure(figsize=(15, 10))

        original_errors = np.sqrt(Y_test[:, 0] ** 2 + Y_test[:, 1] ** 2)

        # Dystrybuanta błędów
        plt.subplot(2, 2, 1)
        sorted_original, cdf_original = AnalysisAndVisualization.calculate_cdf(original_errors)
        plt.plot(sorted_original, cdf_original, label='Błędy oryginalne', linewidth=2, color='red')

        for i, model in enumerate(models):
            predictions = model.predict(X_test)
            corrected_errors = Y_test - predictions
            corrected_magnitude = np.sqrt(corrected_errors[:, 0] ** 2 + corrected_errors[:, 1] ** 2)
            sorted_corrected, cdf_corrected = AnalysisAndVisualization.calculate_cdf(corrected_magnitude)
            plt.plot(sorted_corrected, cdf_corrected, label=f'{model_names[i]}', alpha=0.8)

        plt.xlabel('Błąd [mm]')
        plt.ylabel('Prawdopodobieństwo')
        plt.title('Dystrybuanta błędów')
        plt.legend()
        plt.grid(True)

        # Histogram błędów
        plt.subplot(2, 2, 2)
        plt.hist(original_errors, bins=50, alpha=0.5, label='Oryginalne', density=True, color='red')

        for i, model in enumerate(models):
            predictions = model.predict(X_test)
            corrected_errors = Y_test - predictions
            corrected_magnitude = np.sqrt(corrected_errors[:, 0] ** 2 + corrected_errors[:, 1] ** 2)
            plt.hist(corrected_magnitude, bins=50, alpha=0.6, label=f'{model_names[i]}', density=True)

        plt.xlabel('Błąd [mm]')
        plt.ylabel('Gęstość')
        plt.title('Rozkład błędów')
        plt.legend()
        plt.grid(True)

        # Box plot
        plt.subplot(2, 2, 3)
        error_data = [original_errors]
        labels = ['Oryginalne']

        for i, model in enumerate(models):
            predictions = model.predict(X_test)
            corrected_errors = Y_test - predictions
            corrected_magnitude = np.sqrt(corrected_errors[:, 0] ** 2 + corrected_errors[:, 1] ** 2)
            error_data.append(corrected_magnitude)
            labels.append(model_names[i])

        plt.boxplot(error_data, labels=labels)
        plt.ylabel('Błąd [mm]')
        plt.title('Porównanie rozkładów błędów')
        plt.xticks(rotation=45)
        plt.grid(True)

        # Metryki
        plt.subplot(2, 2, 4)
        metrics_data = []

        for i, model in enumerate(models):
            predictions = model.predict(X_test)
            metrics, _, _ = AnalysisAndVisualization.calculate_error_metrics(Y_test, predictions, Y_test)
            metrics_data.append([
                metrics['mae_original'],
                metrics['mae_corrected'],
                metrics['improvement_percent']
            ])

        metrics_df = pd.DataFrame(metrics_data,
                                  columns=['MAE Original', 'MAE Corrected', 'Improvement %'],
                                  index=model_names)

        x = np.arange(len(model_names))
        width = 0.35

        plt.bar(x - width / 2, metrics_df['MAE Original'], width, label='MAE Original', alpha=0.7)
        plt.bar(x + width / 2, metrics_df['MAE Corrected'], width, label='MAE Corrected', alpha=0.7)

        plt.xlabel('Model')
        plt.ylabel('MAE [mm]')
        plt.title('Porównanie MAE')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Wyświetl szczegółowe metryki
        print("\n=== SZCZEGÓŁOWE METRYKI ===")
        for i, model in enumerate(models):
            predictions = model.predict(X_test)
            metrics, _, _ = AnalysisAndVisualization.calculate_error_metrics(Y_test, predictions, Y_test)
            print(f"\n{model_names[i]}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.3f}")

    @staticmethod
    def plot_outlier_comparison(X, Y, outlier_detector):
        """Porównanie danych przed i po usunięciu outlierów"""
        outliers = outlier_detector.detect_outliers(X, Y)

        plt.figure(figsize=(15, 5))

        # Dane oryginalne
        plt.subplot(1, 3, 1)
        plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.6, s=20)
        plt.scatter(X[outliers, 0], X[outliers, 1], c='red', alpha=0.8, s=20, label='Outliery')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Dane wejściowe z oznaczonymi outlierami')
        plt.legend()
        plt.grid(True)

        # Błędy
        plt.subplot(1, 3, 2)
        errors = np.sqrt(Y[:, 0] ** 2 + Y[:, 1] ** 2)
        plt.scatter(range(len(errors)), errors, c='blue', alpha=0.6, s=20)
        plt.scatter(np.where(outliers)[0], errors[outliers], c='red', alpha=0.8, s=20, label='Outliery')
        plt.xlabel('Indeks próbki')
        plt.ylabel('Błąd [mm]')
        plt.title('Błędy z oznaczonymi outlierami')
        plt.legend()
        plt.grid(True)

        # Histogram błędów
        plt.subplot(1, 3, 3)
        plt.hist(errors[~outliers], bins=50, alpha=0.7, label='Normalne dane', density=True)
        plt.hist(errors[outliers], bins=20, alpha=0.7, label='Outliery', density=True)
        plt.xlabel('Błąd [mm]')
        plt.ylabel('Gęstość')
        plt.title('Rozkład błędów')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        print(f"Wykryto {np.sum(outliers)} outlierów z {len(X)} próbek ({np.sum(outliers) / len(X) * 100:.2f}%)")


