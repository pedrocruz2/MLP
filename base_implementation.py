import numpy as np
import math

class Perceptron:
    def __init__(self, input_size, hidden_size, output_size, activation_threshold=0.5):
        # Inicializar pesos e bias para a camada escondida
        self.hidden_weights = np.random.randn(input_size, hidden_size)
        self.hidden_bias = np.zeros((1, hidden_size))
        
        # Inicializar pesos e bias para a camada de saída
        self.output_weights = np.random.randn(hidden_size, output_size)
        self.output_bias = np.zeros((1, output_size))
        
        self.activation_threshold = activation_threshold

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _activation(self, x):
        return self._sigmoid(x)

    def forward_pass(self, X):
        self.hidden_output = self._activation(np.dot(X, self.hidden_weights) + self.hidden_bias)
        self.output = self._activation(np.dot(self.hidden_output, self.output_weights) + self.output_bias)
        return self.output

    def compute_loss(self, y_true, y_pred):
        # Foi utilizado como função de custo a Cross-Entropy
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backpropagation(self, X, y_true, learning_rate):
        output = self.forward_pass(X)
        
        # Erro da camada de saída
        output_error = y_true - output
        output_delta = output_error * self._sigmoid_derivative(output)
        
        # Erro da camada escondida
        hidden_error = np.dot(output_delta, self.output_weights.T)
        hidden_delta = hidden_error * self._sigmoid_derivative(self.hidden_output)
        
        # Atualização dos pesos e bias da camada de saída
        self.output_weights += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.output_bias += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        # Atualização dos pesos e bias da camada escondida
        self.hidden_weights += np.dot(X.T, hidden_delta) * learning_rate
        self.hidden_bias += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred = self.forward_pass(X)
            loss = self.compute_loss(y, y_pred)
            self.backpropagation(X, y, learning_rate)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        output = self.forward_pass(X)
        return np.round(output)

# Dados de treinamento
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inicialização do perceptron com camada escondida
perceptron = Perceptron(input_size=2, hidden_size=2, output_size=1)

# Treinamento do perceptron
perceptron.train(X, y, epochs=10000, learning_rate=0.1)

# Predição
predictions = perceptron.predict(X)
print("Predições para a função XOR:")
print(predictions)
