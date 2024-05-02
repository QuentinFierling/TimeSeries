from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import pacf
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np


ar2 = np.array([1, 0.72, 0.5])
ma = np.array([1])
simulated_AR2_data = ArmaProcess(ar2, ma).generate_sample(nsample=10000)

plt.figure(figsize=[10, 7.5]); # Set dimensions for figure
plt.plot(simulated_AR2_data)
plt.title("Simulated AR(2) Process")
plt.show()

plot_acf(simulated_AR2_data)

rho, sigma = yule_walker(simulated_AR2_data, 2, method='mle')
print(f'rho: {-rho}')
print(f'sigma: {sigma}')


# Neural Network

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def simulate_ar_process(N, a, n, V0):
    V = [V0]
    for _ in range(n):
        if len(V) <= N:
            # Initialisation aléatoire des premières valeurs avec une loi normale
            V.append(np.random.normal(loc=0, scale=1))
        else:
            # Calcul de la nouvelle valeur en fonction des valeurs précédentes et des coefficients autorégressifs
            new_value = np.dot(V[-N:], a) + np.random.normal(loc=0, scale=1)
            V.append(new_value)
    return V

from numpy.polynomial.polynomial import Polynomial

#Une des conditions pour qu'un processus AR soit stationnaire est que les racines du polynôme caractéristique soient à l'intérieur du cercle unité dans le plan complexe.
def check_stationarity(coefficients):
    # Calcule les racines du polynôme caractéristique
    roots = np.roots(np.flip(np.insert(-coefficients, 0, 1)))  # Ajoute 1 pour le coefficient AR(0)

    # Vérifie si toutes les racines sont à l'intérieur du cercle unité dans le plan complexe
    return all(np.abs(root) < 1 for root in roots)

# Paramètres du processus AR
N = 10  # Ordre du processus AR
n_samples = 10000  # Nombre d'exemples dans le dataset
sequence_length = 200  # Longueur de la séquence de chaque exemple
V0 = 0  # Valeur initiale du processus

# Générer les données du dataset
X = []  # Pour stocker les séquences de données d'entrée (200 points de processus AR)
y = []  # Pour stocker les coefficients AR correspondants à chaque séquence

for _ in range(n_samples):
    a = np.random.uniform(-0.5, 0.5, N)
    if np.sum(np.abs(a)) < 1 and check_stationarity(a):
        break  # Les coefficients respectent la condition, sortie de la boucle
    else:        # Ajustement des coefficients pour respecter la condition
        a /= np.sum(np.abs(a))  # Normalisation des coefficients
    data = simulate_ar_process(N, a, sequence_length, V0)
    # Ajouter la séquence de données d'entrée à X
    X.append(data[:sequence_length])
    # Ajouter les coefficients AR correspondants à y
    y.append(a)

# Convertir X et y en tableaux numpy

#Training data
X_train = torch.tensor(np.array(X[ : int(0.8 * n_samples)]))
y_train = torch.tensor(np.array(y[ : int(0.8 * n_samples)]))

#Test data
X_test = torch.tensor(np.array(X[int(0.8 * n_samples) : ]))
y_test = torch.tensor(np.array(y[int(0.8 * n_samples) : ]))
# Création du DataLoader
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Définition du modèle

class AR_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AR_Net, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        out = self.linear1(x)
        out = self.ReLU(out)
        out = self.linear2(out)
        return out
    
# Initialisation du modèle

input_size = 200  # Taille de la séquence d'entrée
hidden_size = 1056 # Taille de la couche cachée
output_size = N  # Taille de la sortie (nombre de coefficients AR)
model = AR_Net(input_size, hidden_size, output_size)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle

num_epochs = 50
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item()}')
        
print('Training finished')

# Test du modèle

model.eval()
test_output = model(X_test.float())
test_loss = criterion(test_output, y_test.float())
print(f'Test Loss: {test_loss}')
print('True coefficients:', y[0])
print('Predicted coefficients:', test_output[0])



    