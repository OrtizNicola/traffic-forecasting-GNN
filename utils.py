from traffic_data import METRLADatasetLoader 
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# returns the data in the appropiate format
# the loaders with _batch at the end are useful for training
# the loaders with _full at the end are useful to evaluate the model
def get_data_loaders(seed, num_timesteps_in=12, num_timesteps_out=1, train_p=0.8, 
                     batch_size=256, Y_train=None, Y_test=None,
                     pasting=False, bagging=False, pct=0.66):
    loader = METRLADatasetLoader()
    adj, _, x, y, mean, std= loader.get_dataset(num_timesteps_in,
                                                num_timesteps_out)

    X = torch.tensor(np.array(x)).permute(0, 2, 1).unsqueeze(-1)
    Y = torch.tensor(np.array(y))

    train_size = int(train_p * len(X))
    test_size = len(X) - train_size

    X_train, X_test = torch.split(X, [train_size, test_size])

    # uso esto para correr los modelos de boosting, donde se pasaran 
    # los errores para ser ajustados en lugar de los datos originales
    if (Y_train is None) and (Y_test is None):
        Y_train, Y_test = torch.split(Y, [train_size, test_size])

    # en caso de hacer pasting se devuelve un subconjunto aleatorio de los datos
    # en caso de hacer bagging se devulve la misma cantidad de datos pero sampleados
    if pasting or bagging:
        g = torch.Generator()
        g.manual_seed(seed)
        subset_size = pct * X_train.shape[0]

        if pasting == True:
            indices = torch.randperm(X_train.shape[0], generator=g)[:subset_size] # permutacion: no se repiten datos
        elif bagging == True:
            indices = torch.randint(X_train.shape[0], (subset_size,)) # enteros aleatorios: puede haber repeticion

        X_train = X_train[indices]
        Y_train = Y_train[indices]


    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    train_loader_batch = DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True, drop_last=True)
    test_loader_batch = DataLoader(test_dataset, batch_size=batch_size,
                                   shuffle=False, drop_last=True)

    train_loader_full = DataLoader(train_dataset, batch_size=train_size,
                                   shuffle=False, drop_last=False)
    test_loader_full = DataLoader(test_dataset, batch_size=test_size,
                                  shuffle=False, drop_last=False)
    return (train_loader_batch, test_loader_batch, 
            train_loader_full, test_loader_full,
            adj, mean, std)

# normalizes the data as in the DCRNN paper
def transform(X, mean, std):
    Y = X - mean
    Y /= std
    return Y

# brings the data back to the normal values
def inverse_transform(X, mean, std):
    Y = X * std
    Y += mean
    return Y

def train_model(model, device, num_epochs, train_loader, test_loader, 
                adj, lr=0.0001):
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    lossi = []
    lossi_test = []

    # variables utiles para la graficacion
    i = 0
    xs = [0]

    # calculamos la perdida en el conjunto del test antes de entrenar el modelo
    with torch.no_grad():
        model.eval()
        X_batch, Y_batch = next(iter(test_loader))
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        outputs = model(X_batch, adj)
        loss_test = criterion(outputs, Y_batch)
        lossi_test.append(loss_test.item())

    for epoch in range(num_epochs):
        model.train()
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            # Forward
            outputs = model(X_batch, adj)
            loss = criterion(outputs, Y_batch)
            lossi.append(loss.item())

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1

        xs.append(i)

        # calculamos la evolucion de la perdida del test durante el entrenamiento
        with torch.no_grad():
            model.eval()
            X_batch, Y_batch = next(iter(test_loader))
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch, adj)
            loss_test = criterion(outputs, Y_batch)
            lossi_test.append(loss_test.item())

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Loss test: {lossi_test[-1]:.4f}')

        if (lossi_test[-2] < lossi_test[-1]):
            print("Se detectó sobreajuste: Early Stopping!!!")
            print(f"Última época de entrenamiento: {epoch + 1}")
            break
    
    # se devuelve el model, la perdida del entrenamiento, la perdida del test
    # los xs para graficas y los outputs del modelo final en el test
    return (model, lossi, lossi_test, xs, outputs)

def moving_average(data, window_size = 500):
    moving_averages = []
    for i in range(len(data)):
        if i < window_size:
            moving_averages.append(sum(data[:i+1]) / (i+1))
        else:
            moving_averages.append(sum(data[i-window_size+1:i+1]) / window_size)
    return moving_averages
