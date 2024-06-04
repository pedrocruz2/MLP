import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden_output = self.sigmoid(self.hidden_layer(x))
        output = self.sigmoid(self.output_layer(hidden_output))
        return output

# Dados de treinamento
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Inicialização da rede neural
input_size = 2
hidden_size = 2
output_size = 1
model = MLP(input_size, hidden_size, output_size)

# função de custo e otimizador
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Treinamento da rede neural
epochs = 10000
for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Predição
with torch.no_grad():
    predictions = model(X)
    rounded_predictions = torch.round(predictions)
    print("Predições para a função XOR:")
    print(rounded_predictions)

