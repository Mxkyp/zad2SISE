import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


class AnalysisAndVisualization:
    """Klasa do analizy i wizualizacji wyników - kompatybilna z PyTorch"""

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
    def _get_predictions(model, X_test):
        """Uniwersalna funkcja do uzyskiwania predykcji z różnych typów modeli"""
        if hasattr(model, 'predict'):
            # Model typu sklearn/keras
            return model.predict(X_test)
        elif hasattr(model, 'forward') or isinstance(model, torch.nn.Module):
            # Model PyTorch
            model.eval()
            with torch.no_grad():
                if isinstance(X_test, np.ndarray):
                    X_tensor = torch.FloatTensor(X_test)
                else:
                    X_tensor = X_test
                predictions = model(X_tensor)
                return predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
        else:
            # Funkcja callable
            return model(X_test)

    @staticmethod
    def plot_training_history(histories, model_names):
        """Wykres historii trenowania - PyTorch compatible"""
        plt.figure(figsize=(15, 5))

        # Loss
        plt.subplot(1, 3, 1)
        for i, history in enumerate(histories):
            # Sprawdzenie różnych formatów historii
            if hasattr(history, 'history'):
                # Keras format
                hist_dict = history.history
            elif isinstance(history, dict):
                # Słownik z historią
                hist_dict = history
            else:
                # Lista lub inne formaty
                print(f"Nieznany format historii dla modelu {model_names[i]}")
                continue

            # Obsługa różnych nazw kluczy
            loss_key = 'loss' if 'loss' in hist_dict else 'train_loss'
            val_loss_key = 'val_loss' if 'val_loss' in hist_dict else 'valid_loss'

            if loss_key in hist_dict:
                plt.plot(hist_dict[loss_key], label=f'{model_names[i]} - trening')
            if val_loss_key in hist_dict:
                plt.plot(hist_dict[val_loss_key], label=f'{model_names[i]} - walidacja', linestyle='--')

        plt.xlabel('Epoka')
        plt.ylabel('Loss (MSE)')
        plt.title('Historia trenowania - Loss')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)

        # MAE
        plt.subplot(1, 3, 2)
        for i, history in enumerate(histories):
            if hasattr(history, 'history'):
                hist_dict = history.history
            elif isinstance(history, dict):
                hist_dict = history
            else:
                continue

            # Obsługa różnych nazw kluczy dla MAE
            mae_key = 'mae' if 'mae' in hist_dict else 'train_mae'
            val_mae_key = 'val_mae' if 'val_mae' in hist_dict else 'valid_mae'

            if mae_key in hist_dict:
                plt.plot(hist_dict[mae_key], label=f'{model_names[i]} - trening')
            if val_mae_key in hist_dict:
                plt.plot(hist_dict[val_mae_key], label=f'{model_names[i]} - walidacja', linestyle='--')

        plt.xlabel('Epoka')
        plt.ylabel('MAE')
        plt.title('Historia trenowania - MAE')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)

        # Learning Rate (jeśli dostępne)
        plt.subplot(1, 3, 3)
        for i, history in enumerate(histories):
            if hasattr(history, 'history'):
                hist_dict = history.history
            elif isinstance(history, dict):
                hist_dict = history
            else:
                continue

            # Różne nazwy dla learning rate
            lr_keys = ['lr', 'learning_rate', 'LR']
            for lr_key in lr_keys:
                if lr_key in hist_dict:
                    plt.plot(hist_dict[lr_key], label=f'{model_names[i]}')
                    break

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
        """Porównanie błędów dla różnych modeli - PyTorch compatible"""
        plt.figure(figsize=(15, 10))

        original_errors = np.sqrt(Y_test[:, 0] ** 2 + Y_test[:, 1] ** 2)

        # Dystrybuanta błędów
        plt.subplot(2, 2, 1)
        sorted_original, cdf_original = AnalysisAndVisualization.calculate_cdf(original_errors)
        plt.plot(sorted_original, cdf_original, label='Błędy oryginalne', linewidth=2, color='red')

        for i, model in enumerate(models):
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
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
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
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
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
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
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
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
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
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

    @staticmethod
    def plot_model_comparison_detailed(models, X_test, Y_test, model_names):
        """Szczegółowe porównanie modeli z dodatkowymi metrykami - PyTorch compatible"""
        plt.figure(figsize=(20, 12))

        original_errors = np.sqrt(Y_test[:, 0] ** 2 + Y_test[:, 1] ** 2)

        # 1. Wykres błędów w funkcji pozycji
        plt.subplot(3, 3, 1)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=original_errors, cmap='viridis', alpha=0.6, s=30)
        plt.colorbar(label='Błąd [mm]')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Rozkład błędów oryginalnych')
        plt.grid(True)

        # 2-4. Porównanie predykcji każdego modelu
        for i, model in enumerate(models[:3]):  # Maksymalnie 3 modele
            plt.subplot(3, 3, 2 + i)
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
            corrected_errors = Y_test - predictions
            corrected_magnitude = np.sqrt(corrected_errors[:, 0] ** 2 + corrected_errors[:, 1] ** 2)

            plt.scatter(X_test[:, 0], X_test[:, 1], c=corrected_magnitude, cmap='viridis', alpha=0.6, s=30)
            plt.colorbar(label='Błąd [mm]')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Błędy po korekcji - {model_names[i]}')
            plt.grid(True)

        # 5. Porównanie improvement ratio
        plt.subplot(3, 3, 5)
        improvements = []
        for i, model in enumerate(models):
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
            metrics, _, _ = AnalysisAndVisualization.calculate_error_metrics(Y_test, predictions, Y_test)
            improvements.append(metrics['improvement_percent'])

        bars = plt.bar(model_names, improvements, alpha=0.7,
                       color=['skyblue', 'lightgreen', 'lightcoral', 'gold'][:len(models)])
        plt.ylabel('Poprawa [%]')
        plt.title('Procent poprawy MAE')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')

        # Dodanie wartości na słupkach
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{improvement:.1f}%', ha='center', va='bottom')

        # 6. Wykres rozrzutu błędów przed vs po korekcji
        plt.subplot(3, 3, 6)
        for i, model in enumerate(models):
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
            corrected_errors = Y_test - predictions
            corrected_magnitude = np.sqrt(corrected_errors[:, 0] ** 2 + corrected_errors[:, 1] ** 2)

            plt.scatter(original_errors, corrected_magnitude, alpha=0.5, label=model_names[i], s=20)

        # Linia y=x dla referencji
        max_error = max(np.max(original_errors),
                        np.max([np.max(
                            np.sqrt((Y_test - AnalysisAndVisualization._get_predictions(model, X_test))[:, 0] ** 2 +
                                    (Y_test - AnalysisAndVisualization._get_predictions(model, X_test))[:, 1] ** 2))
                                for model in models]))
        plt.plot([0, max_error], [0, max_error], 'k--', alpha=0.5, label='y=x')

        plt.xlabel('Błąd oryginalny [mm]')
        plt.ylabel('Błąd po korekcji [mm]')
        plt.title('Błędy przed vs po korekcji')
        plt.legend()
        plt.grid(True)

        # 7. Percentyle błędów
        plt.subplot(3, 3, 7)
        percentiles = [50, 75, 90, 95, 99]
        x_pos = np.arange(len(percentiles))

        for i, model in enumerate(models):
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
            corrected_errors = Y_test - predictions
            corrected_magnitude = np.sqrt(corrected_errors[:, 0] ** 2 + corrected_errors[:, 1] ** 2)

            model_percentiles = [np.percentile(corrected_magnitude, p) for p in percentiles]
            plt.plot(x_pos, model_percentiles, marker='o', label=model_names[i])

        original_percentiles = [np.percentile(original_errors, p) for p in percentiles]
        plt.plot(x_pos, original_percentiles, marker='s', linestyle='--',
                 label='Oryginalne', color='red', linewidth=2)

        plt.xlabel('Percentyl')
        plt.ylabel('Błąd [mm]')
        plt.title('Percentyle błędów')
        plt.xticks(x_pos, [f'{p}%' for p in percentiles])
        plt.legend()
        plt.grid(True)
        plt.yscale('log')

        # 8. Macierz korelacji błędów
        plt.subplot(3, 3, 8)
        correlation_data = []

        for i, model in enumerate(models):
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
            corrected_errors = Y_test - predictions
            corrected_magnitude = np.sqrt(corrected_errors[:, 0] ** 2 + corrected_errors[:, 1] ** 2)
            correlation_data.append(corrected_magnitude)

        correlation_data.append(original_errors)
        all_names = model_names + ['Oryginalne']

        correlation_matrix = np.corrcoef(correlation_data)

        im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.xticks(range(len(all_names)), all_names, rotation=45)
        plt.yticks(range(len(all_names)), all_names)
        plt.title('Korelacja błędów między modelami')

        # Dodanie wartości korelacji
        for i in range(len(all_names)):
            for j in range(len(all_names)):
                plt.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                         ha='center', va='center', color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')

        # 9. Rozkład błędów relative improvement
        plt.subplot(3, 3, 9)

        for i, model in enumerate(models):
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
            corrected_errors = Y_test - predictions
            corrected_magnitude = np.sqrt(corrected_errors[:, 0] ** 2 + corrected_errors[:, 1] ** 2)

            # Względna poprawa dla każdej próbki
            relative_improvement = (original_errors - corrected_magnitude) / original_errors * 100
            relative_improvement = relative_improvement[np.isfinite(relative_improvement)]  # Usuń inf i nan

            plt.hist(relative_improvement, bins=50, alpha=0.6, label=model_names[i], density=True)

        plt.xlabel('Względna poprawa [%]')
        plt.ylabel('Gęstość')
        plt.title('Rozkład względnej poprawy błędów')
        plt.legend()
        plt.grid(True)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Brak poprawy')

        plt.tight_layout()
        plt.show()

        # Szczegółowy raport liczbowy
        print("\n" + "=" * 80)
        print("SZCZEGÓŁOWY RAPORT PORÓWNAWCZY MODELI")
        print("=" * 80)

        for i, model in enumerate(models):
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
            metrics, _, corrected_magnitude = AnalysisAndVisualization.calculate_error_metrics(Y_test, predictions,
                                                                                               Y_test)

            print(f"\n{model_names[i]}:")
            print(f"  MSE: {metrics['mse_corrected']:.4f} (oryg: {metrics['mse_original']:.4f})")
            print(f"  MAE: {metrics['mae_corrected']:.4f} (oryg: {metrics['mae_original']:.4f})")
            print(f"  RMSE: {metrics['rmse_corrected']:.4f} (oryg: {metrics['rmse_original']:.4f})")
            print(f"  Mediana: {metrics['median_corrected']:.4f} (oryg: {metrics['median_original']:.4f})")
            print(f"  P90: {metrics['p90_corrected']:.4f} (oryg: {metrics['p90_original']:.4f})")
            print(f"  P95: {metrics['p95_corrected']:.4f} (oryg: {metrics['p95_original']:.4f})")
            print(f"  Poprawa MAE: {metrics['improvement_percent']:.2f}%")

            # Dodatkowe statystyki
            successful_corrections = np.sum(corrected_magnitude < original_errors)
            success_rate = successful_corrections / len(corrected_magnitude) * 100
            print(f"  Skuteczność korekcji: {success_rate:.1f}% próbek")

            # Największe poprawy i pogorszenia
            improvements = original_errors - corrected_magnitude
            print(f"  Średnia poprawa: {np.mean(improvements):.4f} mm")
            print(f"  Maks poprawa: {np.max(improvements):.4f} mm")
            print(f"  Maks pogorszenie: {np.min(improvements):.4f} mm")

    @staticmethod
    def plot_prediction_accuracy_map(models, X_test, Y_test, model_names, grid_resolution=50):
        """Mapa dokładności predykcji w przestrzeni wejściowej - PyTorch compatible"""
        plt.figure(figsize=(len(models) * 5, 10))

        # Siatka do interpolacji
        x_min, x_max = X_test[:, 0].min(), X_test[:, 0].max()
        y_min, y_max = X_test[:, 1].min(), X_test[:, 1].max()

        original_errors = np.sqrt(Y_test[:, 0] ** 2 + Y_test[:, 1] ** 2)

        # Oryginalny rozkład błędów
        plt.subplot(2, len(models) + 1, 1)
        scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=original_errors,
                              cmap='viridis', alpha=0.7, s=30)
        plt.colorbar(scatter, label='Błąd [mm]')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Błędy oryginalne')
        plt.grid(True)

        # Dla każdego modelu
        for i, model in enumerate(models):
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
            corrected_errors = Y_test - predictions
            corrected_magnitude = np.sqrt(corrected_errors[:, 0] ** 2 + corrected_errors[:, 1] ** 2)

            # Błędy po korekcji
            plt.subplot(2, len(models) + 1, i + 2)
            scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=corrected_magnitude,
                                  cmap='viridis', alpha=0.7, s=30)
            plt.colorbar(scatter, label='Błąd [mm]')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'{model_names[i]} - Błędy po korekcji')
            plt.grid(True)

            # Względna poprawa
            improvement = (original_errors - corrected_magnitude) / original_errors * 100
            improvement = np.clip(improvement, -100, 100)  # Ograniczenie dla lepszej wizualizacji

            plt.subplot(2, len(models) + 1, len(models) + 2 + i)
            scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=improvement,
                                  cmap='RdYlGn', alpha=0.7, s=30, vmin=-50, vmax=50)
            plt.colorbar(scatter, label='Poprawa [%]')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'{model_names[i]} - Względna poprawa')
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def analyze_model_robustness(models, X_test, Y_test, model_names, noise_levels=[0.01, 0.05, 0.1]):
        """Analiza odporności modeli na szum w danych wejściowych - PyTorch compatible"""
        plt.figure(figsize=(15, 10))

        baseline_errors = []
        for i, model in enumerate(models):
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
            corrected_errors = Y_test - predictions
            corrected_magnitude = np.sqrt(corrected_errors[:, 0] ** 2 + corrected_errors[:, 1] ** 2)
            baseline_errors.append(np.mean(corrected_magnitude))

        # Test odporności na szum
        robustness_data = {name: [] for name in model_names}

        for noise_level in noise_levels:
            for i, model in enumerate(models):
                # Dodanie szumu do danych testowych
                X_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)

                predictions_noisy = AnalysisAndVisualization._get_predictions(model, X_noisy)
                corrected_errors_noisy = Y_test - predictions_noisy
                corrected_magnitude_noisy = np.sqrt(
                    corrected_errors_noisy[:, 0] ** 2 + corrected_errors_noisy[:, 1] ** 2)

                robustness_data[model_names[i]].append(np.mean(corrected_magnitude_noisy))

        # Wykres odporności
        plt.subplot(2, 2, 1)
        for i, name in enumerate(model_names):
            plt.plot([0] + noise_levels, [baseline_errors[i]] + robustness_data[name],
                     marker='o', label=name)

        plt.xlabel('Poziom szumu')
        plt.ylabel('Średni błąd MAE [mm]')
        plt.title('Odporność modeli na szum')
        plt.legend()
        plt.grid(True)

        # Względny wzrost błędu
        plt.subplot(2, 2, 2)
        for i, name in enumerate(model_names):
            relative_increase = [(error - baseline_errors[i]) / baseline_errors[i] * 100
                                 for error in robustness_data[name]]
            plt.plot(noise_levels, relative_increase, marker='o', label=name)

        plt.xlabel('Poziom szumu')
        plt.ylabel('Względny wzrost błędu [%]')
        plt.title('Względny wzrost błędu przy szumie')
        plt.legend()
        plt.grid(True)

        # Test stabilności predykcji (wielokrotne predykcje z szumem)
        plt.subplot(2, 2, 3)
        stability_data = {name: [] for name in model_names}

        test_noise_level = 0.05
        n_runs = 10

        for run in range(n_runs):
            X_noisy = X_test + np.random.normal(0, test_noise_level, X_test.shape)

            for i, model in enumerate(models):
                predictions_noisy = AnalysisAndVisualization._get_predictions(model, X_noisy)
                corrected_errors_noisy = Y_test - predictions_noisy
                corrected_magnitude_noisy = np.sqrt(
                    corrected_errors_noisy[:, 0] ** 2 + corrected_errors_noisy[:, 1] ** 2)
                stability_data[model_names[i]].append(np.mean(corrected_magnitude_noisy))

        # Box plot stabilności
        stability_values = [stability_data[name] for name in model_names]
        plt.boxplot(stability_values, labels=model_names)
        plt.ylabel('MAE [mm]')
        plt.title(f'Stabilność predykcji (szum σ={test_noise_level})')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')

        # Variance analysis
        plt.subplot(2, 2, 4)
        variances = [np.var(stability_data[name]) for name in model_names]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold'][:len(model_names)]
        bars = plt.bar(model_names, variances, color=colors, alpha=0.7)
        plt.ylabel('Wariancja MAE')
        plt.title('Wariancja predykcji przy szumie')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')

        # Dodanie wartości na słupkach
        for bar, var in zip(bars, variances):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{var:.6f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()

        # Raport odporności
        print("\n" + "=" * 60)
        print("RAPORT ODPORNOŚCI MODELI NA SZUM")
        print("=" * 60)

        for i, name in enumerate(model_names):
            print(f"\n{name}:")
            print(f"  Błąd bazowy: {baseline_errors[i]:.6f} mm")

            for j, noise_level in enumerate(noise_levels):
                increase = (robustness_data[name][j] - baseline_errors[i]) / baseline_errors[i] * 100
                print(f"  Szum σ={noise_level}: {robustness_data[name][j]:.6f} mm (+{increase:.1f}%)")

            stability_mean = np.mean(stability_data[name])
            stability_std = np.std(stability_data[name])
            print(f"  Stabilność (μ±σ): {stability_mean:.6f}±{stability_std:.6f} mm")
            print(f"  Współczynnik zmienności: {stability_std / stability_mean * 100:.2f}%")

    @staticmethod
    def plot_residual_analysis(models, X_test, Y_test, model_names):
        """Analiza reszt (residuals) dla modeli - PyTorch compatible"""
        plt.figure(figsize=(len(models) * 5, 12))

        original_errors = np.sqrt(Y_test[:, 0] ** 2 + Y_test[:, 1] ** 2)

        for i, model in enumerate(models):
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
            residuals = Y_test - predictions
            residual_magnitude = np.sqrt(residuals[:, 0] ** 2 + residuals[:, 1] ** 2)

            # Residuals vs Fitted
            plt.subplot(3, len(models), i + 1)
            fitted_magnitude = np.sqrt(predictions[:, 0] ** 2 + predictions[:, 1] ** 2)
            plt.scatter(fitted_magnitude, residual_magnitude, alpha=0.6, s=20)
            plt.xlabel('Wartości dopasowane [mm]')
            plt.ylabel('Reszty [mm]')
            plt.title(f'{model_names[i]} - Reszty vs Dopasowane')
            plt.grid(True)

            # Dodanie linii referencyjnej
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

            # Q-Q plot
            plt.subplot(3, len(models), len(models) + i + 1)
            from scipy import stats
            stats.probplot(residual_magnitude, dist="norm", plot=plt)
            plt.title(f'{model_names[i]} - Q-Q Plot')
            plt.grid(True)

            # Histogram reszt
            plt.subplot(3, len(models), 2 * len(models) + i + 1)
            plt.hist(residual_magnitude, bins=30, density=True, alpha=0.7, color='skyblue')

            # Dodanie krzywej normalnej
            mu, sigma = stats.norm.fit(residual_magnitude)
            x = np.linspace(residual_magnitude.min(), residual_magnitude.max(), 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Rozkład normalny')

            plt.xlabel('Reszty [mm]')
            plt.ylabel('Gęstość')
            plt.title(f'{model_names[i]} - Rozkład reszt')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Testy normalności
        print("\n" + "=" * 50)
        print("TESTY NORMALNOŚCI RESZT")
        print("=" * 50)

        from scipy.stats import shapiro, anderson, jarque_bera

        for i, model in enumerate(models):
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
            residuals = Y_test - predictions
            residual_magnitude = np.sqrt(residuals[:, 0] ** 2 + residuals[:, 1] ** 2)

            print(f"\n{model_names[i]}:")

            # Test Shapiro-Wilka
            if len(residual_magnitude) <= 5000:  # Shapiro działa tylko dla małych próbek
                shapiro_stat, shapiro_p = shapiro(residual_magnitude)
                print(f"  Shapiro-Wilk: stat={shapiro_stat:.4f}, p-value={shapiro_p:.6f}")

            # Test Jarque-Bera
            jb_stat, jb_p = jarque_bera(residual_magnitude)
            print(f"  Jarque-Bera: stat={jb_stat:.4f}, p-value={jb_p:.6f}")

            # Test Andersona-Darlinga
            ad_stat, ad_critical, ad_significance = anderson(residual_magnitude, dist='norm')
            print(f"  Anderson-Darling: stat={ad_stat:.4f}")
            for j, (sig_level, crit_val) in enumerate(zip(ad_significance, ad_critical)):
                result = "ODRZUCAMY" if ad_stat > crit_val else "PRZYJMUJEMY"
                print(f"    {sig_level}%: {crit_val:.4f} -> {result} normalność")

    @staticmethod
    def plot_learning_curves(histories, model_names, metrics=['loss', 'mae']):
        """Szczegółowe krzywe uczenia - PyTorch compatible"""
        n_metrics = len(metrics)
        plt.figure(figsize=(6 * n_metrics, 8))

        for metric_idx, metric in enumerate(metrics):
            # Training curves
            plt.subplot(2, n_metrics, metric_idx + 1)

            for i, history in enumerate(histories):
                if hasattr(history, 'history'):
                    hist_dict = history.history
                elif isinstance(history, dict):
                    hist_dict = history
                else:
                    continue

                # Różne nazwy kluczy
                train_key = metric if metric in hist_dict else f'train_{metric}'
                val_key = f'val_{metric}' if f'val_{metric}' in hist_dict else f'valid_{metric}'

                if train_key in hist_dict:
                    epochs = range(1, len(hist_dict[train_key]) + 1)
                    plt.plot(epochs, hist_dict[train_key], label=f'{model_names[i]} - trening', alpha=0.8)

                    if val_key in hist_dict:
                        plt.plot(epochs, hist_dict[val_key],
                                 label=f'{model_names[i]} - walidacja', linestyle='--', alpha=0.8)

            plt.xlabel('Epoka')
            plt.ylabel(metric.upper())
            plt.title(f'Krzywe uczenia - {metric.upper()}')
            plt.legend()
            plt.grid(True)
            plt.yscale('log' if metric == 'loss' else 'linear')

            # Smoothed curves
            plt.subplot(2, n_metrics, n_metrics + metric_idx + 1)

            for i, history in enumerate(histories):
                if hasattr(history, 'history'):
                    hist_dict = history.history
                elif isinstance(history, dict):
                    hist_dict = history
                else:
                    continue

                train_key = metric if metric in hist_dict else f'train_{metric}'
                val_key = f'val_{metric}' if f'val_{metric}' in hist_dict else f'valid_{metric}'

                if train_key in hist_dict:
                    # Wygładzanie krzywej
                    def smooth_curve(points, factor=0.9):
                        smoothed_points = []
                        for point in points:
                            if smoothed_points:
                                previous = smoothed_points[-1]
                                smoothed_points.append(previous * factor + point * (1 - factor))
                            else:
                                smoothed_points.append(point)
                        return smoothed_points

                    epochs = range(1, len(hist_dict[train_key]) + 1)
                    smoothed_train = smooth_curve(hist_dict[train_key])
                    plt.plot(epochs, smoothed_train, label=f'{model_names[i]} - trening (wygładzone)', alpha=0.8)

                    if val_key in hist_dict:
                        smoothed_val = smooth_curve(hist_dict[val_key])
                        plt.plot(epochs, smoothed_val,
                                 label=f'{model_names[i]} - walidacja (wygładzone)', linestyle='--', alpha=0.8)

            plt.xlabel('Epoka')
            plt.ylabel(metric.upper())
            plt.title(f'Wygładzone krzywe uczenia - {metric.upper()}')
            plt.legend()
            plt.grid(True)
            plt.yscale('log' if metric == 'loss' else 'linear')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def generate_comprehensive_report(models, X_test, Y_test, model_names, histories=None, save_to_file=None):
        """Generowanie komprehensywnego raportu z wszystkich analiz"""
        report = []
        report.append("=" * 80)
        report.append("KOMPREHENSYWNY RAPORT ANALIZY MODELI")
        report.append("=" * 80)
        report.append(f"Data generowania: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Liczba modeli: {len(models)}")
        report.append(f"Rozmiar zbioru testowego: {len(X_test)} próbek")
        report.append("")

        # Podstawowe metryki
        report.append("1. PODSTAWOWE METRYKI JAKOŚCI")
        report.append("-" * 40)

        original_errors = np.sqrt(Y_test[:, 0] ** 2 + Y_test[:, 1] ** 2)

        results_table = []
        for i, model in enumerate(models):
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
            metrics, _, corrected_magnitude = AnalysisAndVisualization.calculate_error_metrics(
                Y_test, predictions, Y_test)

            results_table.append([
                model_names[i],
                f"{metrics['mae_corrected']:.6f}",
                f"{metrics['rmse_corrected']:.6f}",
                f"{metrics['median_corrected']:.6f}",
                f"{metrics['p95_corrected']:.6f}",
                f"{metrics['improvement_percent']:.2f}%"
            ])

        # Tabela wyników
        from tabulate import tabulate
        headers = ['Model', 'MAE [mm]', 'RMSE [mm]', 'Mediana [mm]', 'P95 [mm]', 'Poprawa [%]']
        report.append(tabulate(results_table, headers=headers, tablefmt='grid'))
        report.append("")

        # Ranking modeli
        report.append("2. RANKING MODELI")
        report.append("-" * 20)

        mae_scores = []
        for i, model in enumerate(models):
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
            metrics, _, _ = AnalysisAndVisualization.calculate_error_metrics(Y_test, predictions, Y_test)
            mae_scores.append((model_names[i], metrics['mae_corrected'], metrics['improvement_percent']))

        # Sortowanie według MAE
        mae_scores.sort(key=lambda x: x[1])

        for rank, (name, mae, improvement) in enumerate(mae_scores, 1):
            report.append(f"{rank}. {name}: MAE={mae:.6f} mm, Poprawa={improvement:.2f}%")

        report.append("")

        # Analiza rozkładu błędów
        report.append("3. ANALIZA ROZKŁADU BŁĘDÓW")
        report.append("-" * 30)

        from scipy import stats

        for i, model in enumerate(models):
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
            corrected_errors = Y_test - predictions
            corrected_magnitude = np.sqrt(corrected_errors[:, 0] ** 2 + corrected_errors[:, 1] ** 2)

            # Statystyki opisowe
            report.append(f"\n{model_names[i]}:")
            report.append(f"  Średnia: {np.mean(corrected_magnitude):.6f} mm")
            report.append(f"  Odchylenie standardowe: {np.std(corrected_magnitude):.6f} mm")
            report.append(f"  Skośność: {stats.skew(corrected_magnitude):.4f}")
            report.append(f"  Kurtoza: {stats.kurtosis(corrected_magnitude):.4f}")

            # Percentyle
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            perc_values = np.percentile(corrected_magnitude, percentiles)
            report.append(f"  Percentyle: " +
                          ", ".join([f"P{p}={v:.4f}" for p, v in zip(percentiles, perc_values)]))

        # Historia trenowania (jeśli dostępna)
        if histories:
            report.append("\n4. ANALIZA PROCESU TRENOWANIA")
            report.append("-" * 35)

            for i, history in enumerate(histories):
                if hasattr(history, 'history'):
                    hist_dict = history.history
                elif isinstance(history, dict):
                    hist_dict = history
                else:
                    continue

                report.append(f"\n{model_names[i]}:")

                # Liczba epok
                if 'loss' in hist_dict:
                    report.append(f"  Liczba epok: {len(hist_dict['loss'])}")
                    report.append(f"  Końcowy loss treningowy: {hist_dict['loss'][-1]:.6f}")

                    if 'val_loss' in hist_dict:
                        report.append(f"  Końcowy loss walidacyjny: {hist_dict['val_loss'][-1]:.6f}")

                        # Overfitting check
                        final_gap = hist_dict['val_loss'][-1] - hist_dict['loss'][-1]
                        report.append(f"  Różnica val_loss - train_loss: {final_gap:.6f}")

                        if final_gap > hist_dict['loss'][-1] * 0.1:
                            report.append("  ⚠️  UWAGA: Możliwe przeuczenie!")

        # Rekomendacje
        report.append("\n5. REKOMENDACJE")
        report.append("-" * 15)

        best_model_idx = mae_scores[0]
        report.append(f"• Najlepszy model ogólnie: {best_model_idx[0]}")

        # Model z największą poprawą
        improvement_scores = [(name, improvement) for name, _, improvement in mae_scores]
        improvement_scores.sort(key=lambda x: x[1], reverse=True)
        report.append(f"• Największa poprawa: {improvement_scores[0][0]} ({improvement_scores[0][1]:.2f}%)")

        # Stabilność
        variances = []
        for i, model in enumerate(models):
            predictions = AnalysisAndVisualization._get_predictions(model, X_test)
            corrected_errors = Y_test - predictions
            corrected_magnitude = np.sqrt(corrected_errors[:, 0] ** 2 + corrected_errors[:, 1] ** 2)
            variances.append((model_names[i], np.var(corrected_magnitude)))

        variances.sort(key=lambda x: x[1])
        report.append(f"• Najbardziej stabilny: {variances[0][0]} (wariancja: {variances[0][1]:.8f})")

        report.append("")
        report.append("=" * 80)

        # Łączenie raportu w string
        full_report = "\n".join(report)

        # Zapisanie do pliku jeśli podano
        if save_to_file:
            with open(save_to_file, 'w', encoding='utf-8') as f:
                f.write(full_report)
            print(f"Raport zapisany do: {save_to_file}")

        return full_report