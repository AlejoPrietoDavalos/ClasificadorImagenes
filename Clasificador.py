# Para graficar
import numpy as np
import matplotlib.pyplot as plt

# Las principales de PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# Otras más de PyTorch
from torch.utils.data import DataLoader # Para dividir nuestros datos
from torch.utils.data import sampler # Para muestrar datos
import torchvision.datasets as dataset # Para importar DataSets
import torchvision.transforms as T # Para aplicar transformaciones a nuestros datos
# No es importante, sólo si tienen Jupyter Themes. Nothing to do with Deep Learning
from jupyterthemes import jtplot
jtplot.style()

# Me agrego mi clase para hacer plots.
from Herramientas import UtilsPlot
utilidades = UtilsPlot()


#----------Acá nos fijamos si el procesador tiene CUDA, o sino usamos el CPU----------
dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#----------Acá nos fijamos si el procesador tiene CUDA, o sino usamos el CPU----------






NUM_TRAIN = 55000
BATCH_SIZE = 512

# Get our training, validation and test data.
# data_path = '/media/josh/MyData/Databases/' #use your own data path, you may use an existing data path to avoid having to download the data again.
data_path = 'mnist'     # Elegimos un directorio donde se va a guardar los datos de MNIST en caso de no tenerlos descargados ya.
mnist_train = dataset.MNIST(data_path, train=True, download=True, transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

mnist_val = dataset.MNIST(data_path, train=True, download=True, transform=T.ToTensor())
loader_val = DataLoader(mnist_val, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 60000)))

mnist_test = dataset.MNIST(data_path, train=False, download=True, transform=T.ToTensor())
loader_test = DataLoader(mnist_test, batch_size=BATCH_SIZE)
""" Training Set: Son los datos de entrenamiento y vamos a querer muchos de ellos.
    ###Evaluation Set: No me queda claro con la diferencia del Training Set.
    Test Set: Son datos que la red neuronal NUNCA vió antes, para estimar el error que comete."""



x_test = loader_test.dataset.data
y_test = loader_test.dataset.targets

# Acá graficamos los primeros 25 datos que tenemos en el tensor x_test.
#utilidades.PlotInGrid(5,5,x_test[0:25])


n_inputs = len(torch.flatten(x_test[0]))        # Agarramos la matriz de pixeles de la primer imagen, la convertimos en una lista, y calculamos el largo.
hidden1 = 1000
hidden2 = 1000
output = 10

modelo = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=n_inputs, out_features=hidden1),
    nn.ReLU(),
    #nn.Linear(in_features=hidden1, out_features=hidden2),
    #nn.ReLU(),
    nn.Linear(in_features=hidden2, out_features=output),
)



def compute_acc(loader, model, eval_mode=False):
    num_correct = 0
    num_total = 0
    if eval_mode: model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, pred = scores.max(1)
            assert pred.shape == y.shape, 'Label shape and prediction shape does not match'
            num_correct += (pred==y).sum()
            num_total += pred.size(0)           
            
        return float(num_correct)/num_total


def plot_loss(losses):  
    fig = plt.figure()
    f1 = fig.add_subplot()
    f1.set_ylabel("Cost")
    f1.set_xlabel("Epoch")
    f1.set_title("Cost vs Epoch")
    f1.plot(losses)
    plt.show()
    
def train(model, optimizer, epochs=100):
    model = model.to(device=device)
    losses = []
    
    num_batches = len(loader_train)
    for epoch in range(epochs):
        accum_loss = 0.
        for i, (x, y) in enumerate(loader_train):
            #poner modelo en modo de entrenamiento
            model.train()
            
            #mover a GPU
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            
            #calcular scores
            scores = model(x)
            cost = F.cross_entropy(input=scores, target=y)

            #calcular gradients
            optimizer.zero_grad()
            cost.backward()
            
            #actualizar parametros
            optimizer.step()
            
            #guardar pérdida
            accum_loss += cost.item()
        losses.append(accum_loss / num_batches)
            
        print(f'Epoch: {epoch}, loss: {cost.item()}, val accuracy: {compute_acc(loader_val, model, True)} ')
        print()
    plot_loss(losses)

'''
#entrenar el modelo
learning_rate = 1e-2
epochs = 1
optimizer = torch.optim.SGD(modelo.parameters(), lr=learning_rate)
train(modelo, optimizer, epochs)
'''

import torchvision
img_prueba = torchvision.io.read_image("asd/numero3.png")
print(img_prueba)

modelo.eval()              # turn the model to evaluate mode
with torch.no_grad():     # does not calculate gradient
    class_index = modelo(img_prueba)
print(class_index)




'''
# Para cargar una imagen.
import matplotlib.image as mpimg
img = mpimg.imread("numero3.png")
R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
imgGray = (0.2989 * R + 0.5870 * G + 0.1140 * B)*255            # Fórmula mágica.
imgGray = torch.flatten(torch.tensor(imgGray, dtype=int))       # Lo hacemos tensor.


modelo.eval()              # turn the model to evaluate mode
with torch.no_grad():     # does not calculate gradient
    class_index = modelo(imgGray)#.argmax()   #gets the prediction for the image's class
print(class_index)'''






""" Esto es para ver si tenemos CUDA para procesar en nuestra computadora.
print(torch.cuda.is_available())"""

#utilidades.PlotInGrid(1,1,[x_test[1]])
#print(y_test[1])

#print(y_test.shape)
#print(x_test.shape)
#print(x_test[0])
#print(type(x_test[0]))
#plot_number(x_test[0])


