import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataLoader import DataLoader
from NeutralNetworkModel import EnhancedNeuralNetworkModel
from OutlierDetector import OutlierDetector
import torch
import os


def main():
    """
    Główna funkcja do trenowania i ewaluacji sieci neuronowej
    do korekcji błędów systemu pomiarowego
    """
    print("=== SYSTEM KOREKCJI BŁĘDÓW POMIAROWYCH Z WYKORZYSTANIEM SIECI NEURONOWEJ ===\n")

    # ==============================================
    # 1. ŁADOWANIE I PRZYGOTOWANIE DANYCH
    # ==============================================
    print("1. Ładowanie danych...")

    try:
        # Próba ładowania danych Excel (preferowane)
        static_data, dynamic_data = DataLoader.load_data_excel()

        if static_data is not None and dynamic_data is not None:
            print("✅ Dane Excel załadowane pomyślnie")
            X_train, Y_train, features = DataLoader.prepare_data_excel(static_data)
            X_test, Y_test, _ = DataLoader.prepare_data_excel(dynamic_data)
            data_source = "Excel"
        else:
            raise Exception("Brak danych Excel")

    except Exception as e:
        print(f"⚠️ Nie udało się załadować danych Excel: {e}")
        print("Próba ładowania danych CSV...")

        try:
            # Fallback do danych CSV
            training_data, testing_data = DataLoader.load_data_csv()
            X_train, Y_train, X_test, Y_test = DataLoader.prepare_data_csv(training_data, testing_data)
            data_source = "CSV"
            features = ["input_X", "input_Y"]
            print("✅ Dane CSV załadowane pomyślnie")
        except Exception as csv_e:
            print(f"❌ Błąd ładowania danych CSV: {csv_e}")
            print("Używanie danych syntetycznych do demonstracji...")
            X_train, Y_train, X_test, Y_test = generate_synthetic_data()
            data_source = "Synthetic"
            features = ["synthetic_X", "synthetic_Y"]

    print(f"Źródło danych: {data_source}")
    print(f"Dane treningowe: {X_train.shape}")
    print(f"Dane testowe: {X_test.shape}")
    print(f"Dostępne cechy: {len(features)}")

    # ==============================================
    # 2. KONFIGURACJA MODELI
    # ==============================================
    print("\n2. Konfiguracja modeli...")

    # Model bez eliminacji outlierów (baseline)
    model_baseline = EnhancedNeuralNetworkModel(
        hidden_layers=[128, 64, 32],
        activation_function='relu',
        num_of_inputs_neurons=X_train.shape[1],
        num_of_outputs_neurons=Y_train.shape[1],
        epochs=200,
        learning_rate=0.001,
        optimizer='adam',
        dropout_rate=0.2,
        outlier_detection=False,  # WYŁĄCZONA detekcja outlierów
        batch_size=32,
        id=1
    )

    # Model z eliminacją outlierów (enhanced)
    model_enhanced = EnhancedNeuralNetworkModel(
        hidden_layers=[128, 64, 32],
        activation_function='relu',
        num_of_inputs_neurons=X_train.shape[1],
        num_of_outputs_neurons=Y_train.shape[1],
        epochs=200,
        learning_rate=0.001,
        optimizer='adam',
        dropout_rate=0.2,
        outlier_detection=True,  # WŁĄCZONA detekcja outlierów
        outlier_method='combined',  # Kombinacja metod
        outlier_threshold=3.0,
        batch_size=32,
        id=2
    )

    # Model z alternatywną architekturą
    model_deep = EnhancedNeuralNetworkModel(
        hidden_layers=[256, 128, 64, 32, 16],
        activation_function='relu',
        num_of_inputs_neurons=X_train.shape[1],
        num_of_outputs_neurons=Y_train.shape[1],
        epochs=250,
        learning_rate=0.0005,
        optimizer='adam',
        dropout_rate=0.3,
        outlier_detection=True,
        outlier_method='combined',
        outlier_threshold=2.5,
        batch_size=64,
        id=3
    )

    models = [model_baseline, model_enhanced, model_deep]
    model_names = ['Baseline (bez outlier detection)',
                   'Enhanced (z outlier detection)',
                   'Deep Network (z outlier detection)']

    # ==============================================
    # 3. TRENOWANIE MODELI
    # ==============================================
    print("\n3. Trenowanie modeli...")
    histories = []

    for i, (model, name) in enumerate(zip(models, model_names)):
        print(f"\n--- Trenowanie modelu: {name} ---")
        history = model.train(X_train, Y_train)
        histories.append(history)

        # Zapisanie modelu
        model_filename = f"model_{i + 1}_{name.replace(' ', '_').replace('(', '').replace(')', '')}.pth"
        model.save_weights(model_filename)

    # ==============================================
    # 4. EWALUACJA MODELI
    # ==============================================
    print("\n4. Ewaluacja modeli...")

    # Obliczenie metryk dla każdego modelu
    results = []
    predictions_list = []

    for i, (model, name) in enumerate(zip(models, model_names)):
        print(f"\n--- Ewaluacja modelu: {name} ---")

        # Predykcje
        predictions = model.predict(X_test)
        predictions_list.append(predictions)

        # Błędy przed i po korekcji
        original_errors = Y_test
        corrected_errors = Y_test - predictions

        # Magnitudy błędów
        original_magnitude = np.sqrt(original_errors[:, 0] ** 2 + original_errors[:, 1] ** 2)
        corrected_magnitude = np.sqrt(corrected_errors[:, 0] ** 2 + corrected_errors[:, 1] ** 2)

        # Metryki
        metrics = {
            'model_name': name,
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

        # Poprawa w procentach
        improvement = (metrics['mae_original'] - metrics['mae_corrected']) / metrics['mae_original'] * 100
        metrics['improvement_percent'] = improvement

        results.append(metrics)

        # Wyświetlenie wyników
        print(f"MAE oryginalne: {metrics['mae_original']:.4f} mm")
        print(f"MAE po korekcji: {metrics['mae_corrected']:.4f} mm")
        print(f"Poprawa: {improvement:.2f}%")
        print(f"RMSE oryginalne: {metrics['rmse_original']:.4f} mm")
        print(f"RMSE po korekcji: {metrics['rmse_corrected']:.4f} mm")

    # ==============================================
    # 5. ANALIZA SKUTECZNOŚCI ELIMINACJI OUTLIERÓW
    # ==============================================
    print("\n5. Analiza skuteczności eliminacji outlierów...")

    # Porównanie modelu bez i z eliminacją outlierów
    baseline_mae = results[0]['mae_corrected']
    enhanced_mae = results[1]['mae_corrected']
    outlier_improvement = (baseline_mae - enhanced_mae) / baseline_mae * 100

    print(f"MAE baseline (bez outlier detection): {baseline_mae:.4f} mm")
    print(f"MAE enhanced (z outlier detection): {enhanced_mae:.4f} mm")
    print(f"Poprawa dzięki eliminacji outlierów: {outlier_improvement:.2f}%")

    # Analiza outlierów w danych treningowych
    outlier_detector = OutlierDetector(method='combined', threshold=3.0)
    _, _, clean_indices = outlier_detector.filter_data(X_train, Y_train)
    outlier_ratio = (1 - np.sum(clean_indices) / len(clean_indices)) * 100
    print(f"Odsetek outlierów w danych treningowych: {outlier_ratio:.2f}%")

    # ==============================================
    # 6. WIZUALIZACJA WYNIKÓW
    # ==============================================
    print("\n6. Generowanie wizualizacji...")

    # Historia trenowania
    plot_training_history_custom(histories, model_names)

    # Porównanie błędów
    plot_error_comparison_custom(models, X_test, Y_test, model_names)

    # Dystrybuanty błędów (CDF)
    plot_error_cdf(models, X_test, Y_test, model_names)

    # ==============================================
    # 7. EKSPORT WYNIKÓW
    # ==============================================
    print("\n7. Eksport wyników...")

    # Tabela z metrykami
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("✅ Metryki zapisane do: model_comparison_results.csv")

    # Eksport dystrybuanty najlepszego modelu (zgodnie z wymaganiami)
    best_model_idx = np.argmin([r['mae_corrected'] for r in results])
    best_model = models[best_model_idx]
    best_predictions = best_model.predict(X_test)
    best_corrected_errors = Y_test - best_predictions
    best_corrected_magnitude = np.sqrt(best_corrected_errors[:, 0] ** 2 + best_corrected_errors[:, 1] ** 2)

    # Obliczenie CDF
    sorted_errors = np.sort(best_corrected_magnitude)
    cdf_values = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    # Eksport do Excel (wymaganie zadania)
    cdf_df = pd.DataFrame({'dystrybuanta_bledu': cdf_values})
    cdf_df.to_excel('dystrybuanta_bledu_najlepszy_model.xlsx', index=False)
    print("✅ Dystrybuanta najlepszego modelu zapisana do: dystrybuanta_bledu_najlepszy_model.xlsx")

    # Eksport szczegółowych wyników
    detailed_results = {
        'architecture_info': {
            'baseline': {
                'layers': model_baseline.hidden_layers,
                'activation': model_baseline.activation_function,
                'outlier_detection': False
            },
            'enhanced': {
                'layers': model_enhanced.hidden_layers,
                'activation': model_enhanced.activation_function,
                'outlier_detection': True,
                'outlier_method': 'combined'
            },
            'deep': {
                'layers': model_deep.hidden_layers,
                'activation': model_deep.activation_function,
                'outlier_detection': True,
                'outlier_method': 'combined'
            }
        },
        'training_info': {
            'data_source': data_source,
            'training_samples': X_train.shape[0],
            'testing_samples': X_test.shape[0],
            'input_features': X_train.shape[1],
            'output_features': Y_train.shape[1],
            'outlier_ratio': f"{outlier_ratio:.2f}%"
        },
        'performance_comparison': results
    }

    # Zapis do JSON dla szczegółowej analizy
    import json
    with open('detailed_results.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False, default=str)
    print("✅ Szczegółowe wyniki zapisane do: detailed_results.json")

    # ==============================================
    # 8. PODSUMOWANIE
    # ==============================================
    print("\n" + "=" * 80)
    print("PODSUMOWANIE EKSPERYMENTU")
    print("=" * 80)

    print(f"\n📊 DANE:")
    print(f"  • Źródło: {data_source}")
    print(f"  • Próbki treningowe: {X_train.shape[0]}")
    print(f"  • Próbki testowe: {X_test.shape[0]}")
    print(f"  • Cechy wejściowe: {X_train.shape[1]}")
    print(f"  • Wykryte outliery: {outlier_ratio:.2f}%")

    print(f"\n🏆 NAJLEPSZY MODEL:")
    best_result = results[best_model_idx]
    print(f"  • Nazwa: {best_result['model_name']}")
    print(f"  • MAE przed korekcją: {best_result['mae_original']:.4f} mm")
    print(f"  • MAE po korekcji: {best_result['mae_corrected']:.4f} mm")
    print(f"  • Poprawa: {best_result['improvement_percent']:.2f}%")

    print(f"\n✨ SKUTECZNOŚĆ ELIMINACJI OUTLIERÓW:")
    print(f"  • Poprawa względem baseline: {outlier_improvement:.2f}%")

    print(f"\n📁 WYGENEROWANE PLIKI:")
    print(f"  • model_comparison_results.csv - porównanie metryk")
    print(f"  • dystrybuanta_bledu_najlepszy_model.xlsx - dystrybuanta (wymaganie)")
    print(f"  • detailed_results.json - szczegółowe wyniki")
    print(f"  • model_*.pth - zapisane modele")

    print("\n✅ Eksperyment zakończony pomyślnie!")

    return models, results, histories


def generate_synthetic_data():
    """Generowanie danych syntetycznych do demonstracji"""
    print("Generowanie danych syntetycznych...")

    np.random.seed(42)
    n_train, n_test = 5000, 1000

    # Symulacja błędów systemu pomiarowego
    X_train = np.random.randn(n_train, 2) * 10
    noise_train = np.random.randn(n_train, 2) * 0.5
    Y_train = 0.1 * X_train + noise_train

    X_test = np.random.randn(n_test, 2) * 10
    noise_test = np.random.randn(n_test, 2) * 0.5
    Y_test = 0.1 * X_test + noise_test

    # Dodanie kilku outlierów
    outlier_indices = np.random.choice(n_train, size=int(0.05 * n_train), replace=False)
    Y_train[outlier_indices] += np.random.randn(len(outlier_indices), 2) * 3

    return X_train, Y_train, X_test, Y_test


def plot_training_history_custom(histories, model_names):
    """Wykres historii trenowania"""
    plt.figure(figsize=(15, 5))

    # Loss
    plt.subplot(1, 3, 1)
    for i, history in enumerate(histories):
        if hasattr(history, 'history'):
            hist_dict = history.history
        else:
            hist_dict = history

        if 'loss' in hist_dict:
            plt.plot(hist_dict['loss'], label=f'{model_names[i]} - trening')
        if 'val_loss' in hist_dict:
            plt.plot(hist_dict['val_loss'], label=f'{model_names[i]} - walidacja', linestyle='--')

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
        else:
            hist_dict = history

        if 'mae' in hist_dict:
            plt.plot(hist_dict['mae'], label=f'{model_names[i]} - trening')
        if 'val_mae' in hist_dict:
            plt.plot(hist_dict['val_mae'], label=f'{model_names[i]} - walidacja', linestyle='--')

    plt.xlabel('Epoka')
    plt.ylabel('MAE')
    plt.title('Historia trenowania - MAE')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)

    # Learning Rate
    plt.subplot(1, 3, 3)
    for i, history in enumerate(histories):
        if hasattr(history, 'history'):
            hist_dict = history.history
        else:
            hist_dict = history

        if 'lr' in hist_dict:
            plt.plot(hist_dict['lr'], label=f'{model_names[i]}')

    plt.xlabel('Epoka')
    plt.ylabel('Learning Rate')
    plt.title('Zmiana Learning Rate')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_error_comparison_custom(models, X_test, Y_test, model_names):
    """Porównanie błędów dla różnych modeli"""
    plt.figure(figsize=(15, 10))

    original_errors = np.sqrt(Y_test[:, 0] ** 2 + Y_test[:, 1] ** 2)

    # Box plot
    plt.subplot(2, 2, 1)
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

    # Histogram
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

    # Metryki porównawcze
    plt.subplot(2, 2, 3)
    mae_original = []
    mae_corrected = []

    for model in models:
        predictions = model.predict(X_test)
        corrected_errors = Y_test - predictions
        corrected_magnitude = np.sqrt(corrected_errors[:, 0] ** 2 + corrected_errors[:, 1] ** 2)

        mae_original.append(np.mean(original_errors))
        mae_corrected.append(np.mean(corrected_magnitude))

    x = np.arange(len(model_names))
    width = 0.35

    plt.bar(x - width / 2, mae_original, width, label='MAE Original', alpha=0.7)
    plt.bar(x + width / 2, mae_corrected, width, label='MAE Corrected', alpha=0.7)

    plt.xlabel('Model')
    plt.ylabel('MAE [mm]')
    plt.title('Porównanie MAE')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True)

    # Poprawa w procentach
    plt.subplot(2, 2, 4)
    improvements = [(mae_original[i] - mae_corrected[i]) / mae_original[i] * 100
                    for i in range(len(models))]

    plt.bar(model_names, improvements, alpha=0.7, color='green')
    plt.ylabel('Poprawa [%]')
    plt.title('Poprawa dokładności')
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_error_cdf(models, X_test, Y_test, model_names):
    """Wykres dystrybuanty błędów (CDF) - kluczowy dla zadania"""
    plt.figure(figsize=(12, 8))

    original_errors = np.sqrt(Y_test[:, 0] ** 2 + Y_test[:, 1] ** 2)

    # CDF błędów oryginalnych
    sorted_original = np.sort(original_errors)
    cdf_original = np.arange(1, len(sorted_original) + 1) / len(sorted_original)
    plt.plot(sorted_original, cdf_original, label='Błędy oryginalne',
             linewidth=3, color='red', alpha=0.8)

    # CDF dla każdego modelu
    colors = ['blue', 'green', 'orange', 'purple', 'brown']

    for i, model in enumerate(models):
        predictions = model.predict(X_test)
        corrected_errors = Y_test - predictions
        corrected_magnitude = np.sqrt(corrected_errors[:, 0] ** 2 + corrected_errors[:, 1] ** 2)

        sorted_corrected = np.sort(corrected_magnitude)
        cdf_corrected = np.arange(1, len(sorted_corrected) + 1) / len(sorted_corrected)

        plt.plot(sorted_corrected, cdf_corrected,
                 label=f'{model_names[i]}',
                 linewidth=2, color=colors[i % len(colors)], alpha=0.8)

    plt.xlabel('Błąd [mm]')
    plt.ylabel('Prawdopodobieństwo kumulatywne')
    plt.title('Dystrybuanta błędów pomiaru (CDF)\n' +
              'Im linia jest bardziej przesunięta w lewo, tym lepszy model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, np.percentile(original_errors, 99))  # Ograniczenie do 99 percentyla

    # Dodanie linii percentyli
    percentiles = [50, 90, 95]
    for p in percentiles:
        p_value = np.percentile(original_errors, p)
        plt.axvline(p_value, color='gray', linestyle='--', alpha=0.5)
        plt.text(p_value, 0.1, f'P{p}', rotation=90, alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Ustawienie stylu wykresów
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300

    # Uruchomienie głównej funkcji
    models, results, histories = main()