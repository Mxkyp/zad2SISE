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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
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

        if self.outlier_detection:
            print("Wykonywanie detekcji outlierów...")
            X_clean, Y_clean, clean_indices = self.outlier_detector.outlierDetector(X, Y)
            print(f"Dane po filtracji: {X_clean.shape[0]} próbek")
            return X_clean, Y_clean, clean_indices
        else:
            return X, Y, np.ones(len(X), dtype=bool)

    def train(self, X_train, Y_train, X_val = None, Y_val = None, validation_split=0.2):
        """ Trenowanie sieci neuronowej """
        print("Rozpoczynam trenowanie modelu...")

        # Przetwarzanie wstępne danych treningowych
        X_train_clean, Y_train_clean, _ = self.preprocess_data(X_train, Y_train)

        # Normalizacja danych
        X_train_scaled = self.scaler_X.fit_transform(X_train_clean)
        Y_train_scaled = self.scaler_Y.fit_transform(Y_train_clean)

        # Przygotowanie danych walidacyjnych
        if X_val is not None and Y_val is not None:
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

        # Wybór optymalizatora
        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        else:
            raise ValueError('Optimizer must be either "adam" or "sgd"')

        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7, verbose=True
        )

        # Funkcja straty
        criterion = nn.MSELoss()

        # Early stopping
        best_val_loss = float('inf')
        patience = 20
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
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_mae += torch.mean(torch.abs(outputs - batch_Y)).item()
                num_batches += 1

            # Średnie metryki treningowe
            avg_train_loss = total_loss / num_batches
            avg_train_mae = total_mae / num_batches

            # Walidacja
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping na epoce {epoch + 1}")
                break

            # Scheduler step
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
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()

        predictions = self.scaler_Y.inverse_transform(predictions_scaled)
        return predictions

    def test(self, X_test, Y_test):
        """ Testowanie modelu """
        predictions = self.predict(X_test)
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


