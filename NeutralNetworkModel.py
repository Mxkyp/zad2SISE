import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import OutlierDetector as OutlierDetector


class NeuralNetwork(nn.Module):
    """ Sieć neuronowa """

    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.2, activation='relu'):
        super(NeuralNetwork, self).__init__()

        layers = []

        # Pierwsza warstwa
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        layers.append(nn.Dropout(dropout_rate))

        # Warstwy ukryte
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(dropout_rate))

        # Warstwa wyjściowa
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        self.network = nn.Sequential(*layers)

        # Inicjalizacja wag - POPRAWKA
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            # Używamy Xavier/Glorot initialization zamiast Kaiming dla lepszej stabilności
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)


class EnhancedNeuralNetworkModel:
    """ Klasa modelu sieci neuronowej z mechanizmem eliminacji outlierów """

    def __init__(self, hidden_layers=[128, 64, 32], activation_function='relu',
                 num_of_inputs_neurons=2, num_of_outputs_neurons=2, epochs=200,
                 learning_rate=0.001, optimizer='adam', momentum=0.9, dropout_rate=0.2,
                 outlier_detection=True, outlier_method='combined', outlier_threshold=3.0,
                 batch_size=32, device=None, id=0):

        self.hidden_layers = hidden_layers
        self.activation_function = activation_function
        self.num_of_inputs_neurons = num_of_inputs_neurons
        self.num_of_outputs_neurons = num_of_outputs_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.momentum = momentum
        self.dropout_rate = dropout_rate
        self.outlier_threshold = outlier_threshold
        self.outlier_detection = outlier_detection
        self.batch_size = batch_size
        self.id = id

        # Ustawienie urządzeinia (GPU/CPU)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Używane urządzenie: {self.device}")

        # Inicjalizacja detektora outlierów
        if self.outlier_detection:
            self.outlier_detector = OutlierDetector.OutlierDetector(method=outlier_method, threshold=outlier_threshold)

        self.model = self.create_model()
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': [], 'lr': []}

    def create_model(self):
        """ Tworzenie architektury sieci neuronowej """
        model = NeuralNetwork(input_size=self.num_of_inputs_neurons,
                              hidden_layers=self.hidden_layers,
                              output_size=self.num_of_outputs_neurons,
                              dropout_rate=self.dropout_rate,
                              activation=self.activation_function).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model utworzony ✅ - Warstwy: {self.hidden_layers}")
        print(f"Łączna liczba parametrów: {total_params}")
        print(f"Parametry treningowe: {trainable_params}")
        return model

    def preprocess_data(self, X, Y):
        """ Przetwarzanie wstępne danych z opcjonalną eliminacją outlierów """
        print(f"Dane wejściowe: {X.shape[0]} próbek")

        # POPRAWKA: Sprawdzenie i usunięcie NaN/inf przed przetwarzaniem
        print("Sprawdzanie NaN/inf w danych...")

        # Sprawdzenie X
        X_nan_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
        Y_nan_mask = np.isnan(Y).any(axis=1) | np.isinf(Y).any(axis=1)

        # Kombinacja masek
        clean_mask = ~(X_nan_mask | Y_nan_mask)

        if not clean_mask.all():
            print(f"Usunięto {(~clean_mask).sum()} próbek z NaN/inf")
            X = X[clean_mask]
            Y = Y[clean_mask]

        if self.outlier_detection:
            print("Wykonywanie detekcji outlierów...")
            X_clean, Y_clean, clean_indices = self.outlier_detector.outlierDetector(X, Y)
            print(f"Dane po filtracji: {X_clean.shape[0]} próbek")
            return X_clean, Y_clean, clean_indices
        else:
            return X, Y, np.ones(len(X), dtype=bool)

    def train(self, X_train, Y_train, X_val=None, Y_val=None, validation_split=0.2):
        """ Trenowanie sieci neuronowej """
        print("Rozpoczynam trenowanie modelu...")

        # Przetwarzanie wstępne danych treningowych
        X_train_clean, Y_train_clean, _ = self.preprocess_data(X_train, Y_train)

        # POPRAWKA: Dodatkowe sprawdzenie po preprocessingu
        if len(X_train_clean) == 0:
            raise ValueError("Brak danych po preprocessingu!")

        # Normalizacja danych
        X_train_scaled = self.scaler_X.fit_transform(X_train_clean)
        Y_train_scaled = self.scaler_Y.fit_transform(Y_train_clean)

        # POPRAWKA: Sprawdzenie czy normalizacja nie wprowadziła NaN
        if np.isnan(X_train_scaled).any() or np.isnan(Y_train_scaled).any():
            print("UWAGA: NaN po normalizacji - używam robust scaler")
            from sklearn.preprocessing import RobustScaler
            self.scaler_X = RobustScaler()
            self.scaler_Y = RobustScaler()
            X_train_scaled = self.scaler_X.fit_transform(X_train_clean)
            Y_train_scaled = self.scaler_Y.fit_transform(Y_train_clean)

        # Przygotowanie danych walidacyjnych
        if X_val is not None and Y_val is not None:
            # POPRAWKA: Sprawdzenie danych walidacyjnych
            val_nan_mask = np.isnan(X_val).any(axis=1) | np.isinf(X_val).any(axis=1)
            val_y_nan_mask = np.isnan(Y_val).any(axis=1) | np.isinf(Y_val).any(axis=1)
            val_clean_mask = ~(val_nan_mask | val_y_nan_mask)

            if not val_clean_mask.all():
                print(f"Usunięto {(~val_clean_mask).sum()} próbek walidacyjnych z NaN/inf")
                X_val = X_val[val_clean_mask]
                Y_val = Y_val[val_clean_mask]

            X_val_scaled = self.scaler_X.transform(X_val)
            Y_val_scaled = self.scaler_Y.transform(Y_val)
        else:
            # Podział na treningowe i walidacyjne
            split_idx = int(len(X_train_scaled) * (1 - validation_split))
            X_val_scaled = X_train_scaled[split_idx:]
            Y_val_scaled = Y_train_scaled[split_idx:]
            X_train_scaled = X_train_scaled[:split_idx]
            Y_train_scaled = Y_train_scaled[:split_idx]

        # Konwersja do tensorów PyTorch
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        Y_train_tensor = torch.FloatTensor(Y_train_scaled).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        Y_val_tensor = torch.FloatTensor(Y_val_scaled).to(self.device)

        # DataLoader
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Wybór optymalizatora - POPRAWKA: Niższy learning rate
        effective_lr = min(self.learning_rate, 0.001)  # Ograniczenie learning rate

        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=effective_lr, weight_decay=1e-5)
        elif self.optimizer_type == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=effective_lr, momentum=self.momentum, weight_decay=1e-5)
        else:
            raise ValueError('Optimizer must be either "adam" or "sgd"')

        # Scheduler - POPRAWKA: Bardziej konserwatywny scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=15, min_lr=1e-8
        )

        # Funkcja straty
        criterion = nn.MSELoss()

        # Early stopping
        best_val_loss = float('inf')
        patience = 25  # POPRAWKA: Zwiększona cierpliwość
        patience_counter = 0
        best_model_state = None

        # Trenowanie
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            total_mae = 0
            num_batches = 0

            for batch_X, batch_Y in train_loader:
                optimizer.zero_grad()

                # POPRAWKA: Gradient clipping
                outputs = self.model(batch_X)

                # Sprawdzenie czy outputs zawiera NaN
                if torch.isnan(outputs).any():
                    print(f"NaN w outputs na epoce {epoch + 1}!")
                    break

                loss = criterion(outputs, batch_Y)

                # Sprawdzenie czy loss jest NaN
                if torch.isnan(loss):
                    print(f"NaN w loss na epoce {epoch + 1}!")
                    break

                loss.backward()

                # POPRAWKA: Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()
                total_mae += torch.mean(torch.abs(outputs - batch_Y)).item()
                num_batches += 1

            # Średnie metryki treningowe
            avg_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            avg_train_mae = total_mae / num_batches if num_batches > 0 else float('inf')

            # Walidacja
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)

                # POPRAWKA: Sprawdzenie NaN w walidacji
                if torch.isnan(val_outputs).any():
                    print(f"NaN w walidacji na epoce {epoch + 1}!")
                    val_loss = float('inf')
                    val_mae = float('inf')
                else:
                    val_loss = criterion(val_outputs, Y_val_tensor).item()
                    val_mae = torch.mean(torch.abs(val_outputs - Y_val_tensor)).item()
            self.model.train()

            # Zapisywanie historii
            self.history['loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['mae'].append(avg_train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])

            # Early stopping
            if val_loss < best_val_loss and not np.isnan(val_loss) and not np.isinf(val_loss):
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping na epoce {epoch + 1}")
                break

            # Scheduler step
            if not np.isnan(val_loss) and not np.isinf(val_loss):
                scheduler.step(val_loss)

            # Wyświetlanie postępu
            if (epoch + 1) % 10 == 0:
                print(f"Epoka {epoch + 1}/{self.epochs} - "
                      f"Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                      f"MAE: {avg_train_mae:.6f}, Val MAE: {val_mae:.6f}")

        # Przywrócenie najlepszego modelu
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        print("Model wytrenowany ✅")

        # Tworzenie obiektu historii kompatybilnego z kodem analizy
        class HistoryWrapper:
            def __init__(self, history_dict):
                self.history = history_dict

        return HistoryWrapper(self.history)

    def predict(self, X):
        """ Predykcja korekcji błędów """
        # POPRAWKA: Sprawdzenie danych wejściowych
        if np.isnan(X).any() or np.isinf(X).any():
            print("UWAGA: NaN/inf w danych do predykcji - filtrowanie...")
            clean_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
            if not clean_mask.any():
                raise ValueError("Wszystkie dane do predykcji zawierają NaN/inf!")
            X_clean = X[clean_mask]
        else:
            X_clean = X
            clean_mask = np.ones(len(X), dtype=bool)

        X_scaled = self.scaler_X.transform(X_clean)

        # Sprawdzenie po skalowaniu
        if np.isnan(X_scaled).any():
            print("NaN po skalowaniu - problem ze scalerem!")
            raise ValueError("NaN wprowadzone przez scaler!")

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()

            # POPRAWKA: Sprawdzenie predykcji
            if np.isnan(predictions_scaled).any():
                print("NaN w predykcjach modelu!")
                # Zastąpienie NaN zerami jako fallback
                predictions_scaled = np.nan_to_num(predictions_scaled, nan=0.0)

        predictions_clean = self.scaler_Y.inverse_transform(predictions_scaled)

        # Rekonstrukcja pełnego wektora predykcji
        if not clean_mask.all():
            predictions = np.zeros((len(X), predictions_clean.shape[1]))
            predictions[clean_mask] = predictions_clean
        else:
            predictions = predictions_clean

        return predictions

    def test(self, X_test, Y_test):
        """ Testowanie modelu """
        predictions = self.predict(X_test)

        # POPRAWKA: Sprawdzenie przed obliczeniem MSE
        if np.isnan(predictions).any():
            print("UWAGA: NaN w predykcjach - zastępowanie zerami")
            predictions = np.nan_to_num(predictions, nan=0.0)

        mse = mean_squared_error(Y_test, predictions)
        return mse, predictions

    def save_weights(self, filename):
        """ Zapisywanie wag modelu """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_Y': self.scaler_Y,
            'model_config': {
                'hidden_layers': self.hidden_layers,
                'activation_function': self.activation_function,
                'num_of_inputs_neurons': self.num_of_inputs_neurons,
                'num_of_outputs_neurons': self.num_of_outputs_neurons,
                'dropout_rate': self.dropout_rate,
            }
        }, filename)
        print(f"Model zapisany do {filename}")

    def load_weights(self, filename):
        """ Ładowanie wag modelu """
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_Y = checkpoint['scaler_Y']
        print(f"Model załadowany z {filename}")

    def save_weights_csv(self, filename):
        """ Zapisywanie wag modelu do pliku CSV """
        weights_data = []
        layer_info = []

        for name, param in self.model.named_parameters():
            weights_data.append(param.detach().cpu().numpy().flatten())
            layer_info.append(name)

        all_weights = np.concatenate(weights_data)
        weights_df = pd.DataFrame({'weights': all_weights})
        weights_df.to_csv(filename, index=False)
        print(f"Wagi zapisane do {filename}")