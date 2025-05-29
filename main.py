import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DataLoader import DataLoader
from NeutralNetworkModel import EnhancedNeuralNetworkModel


def plot_training_history(history):
    """Wykres historii trenowania"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoka')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)

    # Learning Rate
    ax3.plot(history.history['lr'])
    ax3.set_title('Learning Rate')
    ax3.set_xlabel('Epoka')
    ax3.set_ylabel('Learning Rate')
    ax3.grid(True)

    # Ostatni wykres - pusty lub dodatkowe metryki
    ax4.text(0.5, 0.5, 'Dodatkowe metryki\nmogą być tutaj',
             ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_distributions(Y_test_original, Y_test_corrected, title_suffix=""):
    """Porównanie rozkładów błędów przed i po korekcji"""

    # Obliczenie błędów absolutnych
    errors_original = np.sqrt(Y_test_original[:, 0] ** 2 + Y_test_original[:, 1] ** 2)
    errors_corrected = np.sqrt(Y_test_corrected[:, 0] ** 2 + Y_test_corrected[:, 1] ** 2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram błędów
    ax1.hist(errors_original, bins=50, alpha=0.7, label='Przed korekcją', density=True)
    ax1.hist(errors_corrected, bins=50, alpha=0.7, label='Po korekcji', density=True)
    ax1.set_xlabel('Błąd [m]')
    ax1.set_ylabel('Gęstość')
    ax1.set_title(f'Rozkład błędów {title_suffix}')
    ax1.legend()
    ax1.grid(True)

    # Dystrybuanta CDF
    sorted_orig = np.sort(errors_original)
    sorted_corr = np.sort(errors_corrected)
    cdf_orig = np.arange(1, len(sorted_orig) + 1) / len(sorted_orig)
    cdf_corr = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr)

    ax2.plot(sorted_orig, cdf_orig, label='Przed korekcją', linewidth=2)
    ax2.plot(sorted_corr, cdf_corr, label='Po korekcji', linewidth=2)
    ax2.set_xlabel('Błąd [m]')
    ax2.set_ylabel('Dystrybuanta F(x)')
    ax2.set_title(f'Dystrybuanta błędów {title_suffix}')
    ax2.legend()
    ax2.grid(True)

    # Scatter plot błędów X vs Y
    ax3.scatter(Y_test_original[:, 0], Y_test_original[:, 1], alpha=0.5, label='Przed korekcją', s=1)
    ax3.scatter(Y_test_corrected[:, 0], Y_test_corrected[:, 1], alpha=0.5, label='Po korekcji', s=1)
    ax3.set_xlabel('Błąd X [m]')
    ax3.set_ylabel('Błąd Y [m]')
    ax3.set_title(f'Błędy X vs Y {title_suffix}')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')

    plt.tight_layout()
    plt.savefig(f'error_analysis_{title_suffix.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return sorted_corr, cdf_corr


def calculate_metrics(Y_true, Y_pred):
    """Obliczenie metryk błędów"""
    errors = np.sqrt(Y_true[:, 0] ** 2 + Y_true[:, 1] ** 2)
    errors_pred = np.sqrt(Y_pred[:, 0] ** 2 + Y_pred[:, 1] ** 2)

    metrics = {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        'max_error': np.max(errors),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mae': np.mean(errors),
        'q95': np.percentile(errors, 95),
        'q99': np.percentile(errors, 99)
    }

    return metrics, errors


def save_cdf_to_excel(errors, filename):
    """Zapisanie dystrybuanty do pliku Excel"""
    sorted_errors = np.sort(errors)
    df = pd.DataFrame({'error': sorted_errors})
    df.to_excel(filename, index=False)
    print(f"Dystrybuanta zapisana do {filename}")


def main():
    print("=== SYSTEM KOREKCJI BŁĘDÓW UWB Z SIECIĄ NEURONOWĄ ===\n")

    try:
        # 1. Ładowanie i przygotowanie danych
        print("1. Ładowanie danych...")
        X_train, Y_train, X_test, Y_test, features = DataLoader.prepare_training_testing_data()

        print(f"\nCechy wejściowe ({len(features)}):")
        for i, feature in enumerate(features):
            print(f"  {i + 1}. {feature}")

        # 2. Konfiguracja modeli
        print("\n2. Konfiguracja modeli...")

        # Model bez eliminacji outlierów
        model_without_outlier = EnhancedNeuralNetworkModel(
            hidden_layers=[128, 64, 32],
            activation_function='relu',
            num_of_inputs_neurons=X_train.shape[1],
            num_of_outputs_neurons=2,
            epochs=200,
            learning_rate=0.001,
            outlier_detection=False,
            batch_size=64,
            id=1
        )

        # Model z eliminacją outlierów
        model_with_outlier = EnhancedNeuralNetworkModel(
            hidden_layers=[128, 64, 32],
            activation_function='relu',
            num_of_inputs_neurons=X_train.shape[1],
            num_of_outputs_neurons=2,
            epochs=200,
            learning_rate=0.001,
            outlier_detection=True,
            outlier_method='combined',
            outlier_threshold=3.0,
            batch_size=64,
            id=2
        )

        # 3. Trenowanie modelu bez eliminacji outlierów
        print("\n3. Trenowanie modelu BEZ eliminacji outlierów...")
        history1 = model_without_outlier.train(X_train, Y_train)

        # 4. Trenowanie modelu z eliminacją outlierów
        print("\n4. Trenowanie modelu Z eliminacją outlierów...")
        history2 = model_with_outlier.train(X_train, Y_train)

        # 5. Testowanie modeli
        print("\n5. Testowanie modeli...")

        # Test modelu bez outlierów
        mse1, predictions1 = model_without_outlier.test(X_test, Y_test)
        Y_corrected1 = Y_test - predictions1  # Korekcja błędów

        # Test modelu z outlierami
        mse2, predictions2 = model_with_outlier.test(X_test, Y_test)
        Y_corrected2 = Y_test - predictions2  # Korekcja błędów

        print(f"\nMSE bez eliminacji outlierów: {mse1:.6f}")
        print(f"MSE z eliminacją outlierów: {mse2:.6f}")

        # 6. Analiza wyników
        print("\n6. Analiza wyników...")

        # Metryki oryginalnych błędów
        metrics_orig, errors_orig = calculate_metrics(Y_test, Y_test)
        print("\nMetryki oryginalnych błędów:")
        for key, value in metrics_orig.items():
            print(f"  {key}: {value:.4f}")

        # Metryki po korekcji (bez outlierów)
        metrics_corr1, errors_corr1 = calculate_metrics(Y_corrected1, Y_corrected1)
        print("\nMetryki po korekcji (bez eliminacji outlierów):")
        for key, value in metrics_corr1.items():
            print(f"  {key}: {value:.4f}")

        # Metryki po korekcji (z outlierami)
        metrics_corr2, errors_corr2 = calculate_metrics(Y_corrected2, Y_corrected2)
        print("\nMetryki po korekcji (z eliminacją outlierów):")
        for key, value in metrics_corr2.items():
            print(f"  {key}: {value:.4f}")

        # 7. Wizualizacje
        print("\n7. Generowanie wykresów...")

        # Historia trenowania
        plot_training_history(history1)
        plot_training_history(history2)

        # Porównanie rozkładów błędów
        cdf_orig, _ = plot_error_distributions(Y_test, Y_test, "- Dane oryginalne")
        cdf_corr1, _ = plot_error_distributions(Y_test, Y_corrected1, "- Bez eliminacji outlierów")
        cdf_corr2, _ = plot_error_distributions(Y_test, Y_corrected2, "- Z eliminacją outlierów")

        # 8. Zapis wyników
        print("\n8. Zapisywanie wyników...")

        # Zapisanie dystrybuant do Excel
        save_cdf_to_excel(errors_orig, 'dystrybuanta_oryginalna.xlsx')
        save_cdf_to_excel(errors_corr1, 'dystrybuanta_bez_outlierow.xlsx')
        save_cdf_to_excel(errors_corr2, 'dystrybuanta_z_outlierami.xlsx')

        # Zapisanie modeli
        model_without_outlier.save_weights('model_bez_outlierow.pth')
        model_with_outlier.save_weights('model_z_outlierami.pth')

        # Zapisanie szczegółowych wyników
        results_df = pd.DataFrame({
            'original_error_x': Y_test[:, 0],
            'original_error_y': Y_test[:, 1],
            'corrected_error_x_no_outlier': Y_corrected1[:, 0],
            'corrected_error_y_no_outlier': Y_corrected1[:, 1],
            'corrected_error_x_with_outlier': Y_corrected2[:, 0],
            'corrected_error_y_with_outlier': Y_corrected2[:, 1],
            'prediction_x_no_outlier': predictions1[:, 0],
            'prediction_y_no_outlier': predictions1[:, 1],
            'prediction_x_with_outlier': predictions2[:, 0],
            'prediction_y_with_outlier': predictions2[:, 1]
        })
        results_df.to_excel('wyniki_szczegolowe.xlsx', index=False)

        # 9. Generowanie raportu końcowego
        print("\n9. Generowanie raportu końcowego...")

        # Zapisanie podsumowania do pliku tekstowego
        with open('raport_podsumowanie.txt', 'w', encoding='utf-8') as f:
            f.write("=== RAPORT KOŃCOWY - SYSTEM KOREKCJI BŁĘDÓW UWB ===\n\n")

            f.write("1. ARCHITEKTURA SIECI NEURONOWEJ:\n")
            f.write(f"   - Liczba warstw ukrytych: {len(model_without_outlier.hidden_layers)}\n")
            f.write(f"   - Neurony w warstwach: {model_without_outlier.hidden_layers}\n")
            f.write(f"   - Funkcja aktywacji: {model_without_outlier.activation_function}\n")
            f.write(f"   - Neurony wejściowe: {model_without_outlier.num_of_inputs_neurons}\n")
            f.write(f"   - Neurony wyjściowe: {model_without_outlier.num_of_outputs_neurons}\n")
            f.write(f"   - Dropout rate: {model_without_outlier.dropout_rate}\n")
            f.write(f"   - Batch size: {model_without_outlier.batch_size}\n")
            f.write(f"   - Learning rate: {model_without_outlier.learning_rate}\n")
            f.write(f"   - Liczba epok: {model_without_outlier.epochs}\n\n")

            f.write("2. DANE TRENINGOWE I TESTOWE:\n")
            f.write(f"   - Próbki treningowe: {X_train.shape[0]}\n")
            f.write(f"   - Próbki testowe: {X_test.shape[0]}\n")
            f.write(f"   - Liczba cech: {X_train.shape[1]}\n\n")

            f.write("3. CECHY WEJŚCIOWE:\n")
            for i, feature in enumerate(features):
                f.write(f"   {i + 1}. {feature}\n")
            f.write("\n")

            f.write("4. WYNIKI MODELU BEZ ELIMINACJI OUTLIERÓW:\n")
            f.write(f"   - MSE: {mse1:.6f}\n")
            for key, value in metrics_corr1.items():
                f.write(f"   - {key}: {value:.4f}\n")
            f.write("\n")

            f.write("5. WYNIKI MODELU Z ELIMINACJĄ OUTLIERÓW:\n")
            f.write(f"   - MSE: {mse2:.6f}\n")
            for key, value in metrics_corr2.items():
                f.write(f"   - {key}: {value:.4f}\n")
            f.write("\n")

            f.write("6. PORÓWNANIE WYNIKÓW:\n")
            improvement1 = ((metrics_orig['mae'] - metrics_corr1['mae']) / metrics_orig['mae'] * 100)
            improvement2 = ((metrics_orig['mae'] - metrics_corr2['mae']) / metrics_orig['mae'] * 100)
            f.write(f"   - Poprawa błędu (bez eliminacji outlierów): {improvement1:.2f}%\n")
            f.write(f"   - Poprawa błędu (z eliminacją outlierów): {improvement2:.2f}%\n")
            f.write(f"   - Różnica między modelami: {improvement2 - improvement1:.2f}%\n\n")

            f.write("7. MECHANIZM ELIMINACJI OUTLIERÓW:\n")
            f.write("   - Metoda: Kombinacja trzech algorytmów (Z-score, IQR, Mahalanobis)\n")
            f.write("   - Próg outliera: >= 2 metody muszą wykryć punkt jako outlier\n")
            f.write("   - Automatyczna eliminacja przed treningiem sieci\n\n")

            f.write("8. PLIKI WYGENEROWANE:\n")
            f.write("   - dystrybuanta_oryginalna.xlsx\n")
            f.write("   - dystrybuanta_bez_outlierow.xlsx\n")
            f.write("   - dystrybuanta_z_outlierami.xlsx\n")
            f.write("   - model_bez_outlierow.pth\n")
            f.write("   - model_z_outlierami.pth\n")
            f.write("   - wyniki_szczegolowe.xlsx\n")
            f.write("   - training_history.png (x2)\n")
            f.write("   - error_analysis_*.png (x3)\n")

        print("\n=== ZAKOŃCZONO POMYŚLNIE ===")
        print(
            f"Poprawa błędu (bez eliminacji outlierów): {((metrics_orig['mae'] - metrics_corr1['mae']) / metrics_orig['mae'] * 100):.2f}%")
        print(
            f"Poprawa błędu (z eliminacją outlierów): {((metrics_orig['mae'] - metrics_corr2['mae']) / metrics_orig['mae'] * 100):.2f}%")

        print("\n=== PLIKI WYGENEROWANE ===")
        print("📊 Dystrybuanty błędów:")
        print("   - dystrybuanta_oryginalna.xlsx")
        print("   - dystrybuanta_bez_outlierow.xlsx")
        print("   - dystrybuanta_z_outlierami.xlsx")
        print("🤖 Modele sieci neuronowych:")
        print("   - model_bez_outlierow.pth")
        print("   - model_z_outlierami.pth")
        print("📈 Wykresy i analizy:")
        print("   - training_history.png")
        print("   - error_analysis_*.png")
        print("📋 Raporty:")
        print("   - wyniki_szczegolowe.xlsx")
        print("   - raport_podsumowanie.txt")

    except FileNotFoundError as e:
        print(f"\n❌ BŁĄD: Nie znaleziono pliku: {str(e)}")
        print("Sprawdź czy struktura katalogów jest poprawna:")
        print("  - /dane/F8/*.xlsx")
        print("  - /dane/F10/*.xlsx")

    except ImportError as e:
        print(f"\n❌ BŁĄD IMPORTU: {str(e)}")
        print("Zainstaluj wymagane biblioteki:")
        print("  pip install torch pandas numpy matplotlib scikit-learn scipy openpyxl")

    except Exception as e:
        print(f"\n❌ NIEOCZEKIWANY BŁĄD: {str(e)}")
        import traceback
        print("\nPełny stack trace:")
        traceback.print_exc()

        print("\n🔧 MOŻLIWE ROZWIĄZANIA:")
        print("1. Sprawdź czy wszystkie pliki .xlsx znajdują się w odpowiednich katalogach")
        print("2. Upewnij się, że pliki nie są uszkodzone")
        print("3. Sprawdź czy masz wystarczającą ilość RAM (zalecane >4GB)")
        print("4. Sprawdź czy Python ma uprawnienia do zapisu w bieżącym katalogu")


if __name__ == "__main__":
    main()