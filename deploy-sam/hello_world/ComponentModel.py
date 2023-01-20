import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

def LinearCell(num_input, num_hidden, Dropout=0):
    Seq = nn.Sequential(
        nn.Linear(num_input,num_hidden),
        nn.LeakyReLU(0.8),
        nn.Dropout(Dropout)
    )
    return Seq

class Net(nn.Module):
    def __init__(self,input_size: int =15,hidden_size: int=18,output_size: int=15,num_layers:int =3, num_linear:int =4, from_file :bool= False, file_path : str= None):
        """
        input_size: Cantidad de componentes
        hidden_size: Cantidad de neuronas en la capa oculta
        output_size: Cantidad de componentes
        num_layers: Cantidad de capas ocultas para LSTM
        num_linear: Cantidad de capas ocultas para MLP
        from_file: Cargar modelo desde archivo
        file_path: Ruta del archivo
        """
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers)
        LinearSeq = []
        for i in range(num_linear):
            LinearSeq.append(LinearCell(hidden_size,hidden_size,Dropout=0))
        self.LinearSeq = nn.Sequential(*LinearSeq)
        self.LOut = LinearCell(hidden_size,output_size)
        self.init_weights()
        if from_file:
            self.load_state_dict(torch.load(file_path))
            #print ("Model loaded from file...Cimplete")
    def init_weights(self):
        """
        Inicializa los pesos de las capas lineales
        | Metodo de Xavier |
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.bias)

    def ini_hidden(self, batch_size : int = 1):
        """
        |Devuelve el h_t y c_t inicializados en ceros|
        batch_size: Tamaño de la muestra [Sugerible de 1]
        h_t: Estado oculto [num_layers,batch_size,hidden_size]
        c_t: Estado de memoria [num_layers,batch_size,hidden_size]
        """
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
    
    def forward(self,x : Tensor ,h_t:Tensor=None, c_t:Tensor=None, future:int = 0,with_iteration:bool = False) -> torch.Tensor:
        """
        x: input [,batch_size,input_size]
        x: input [Longitud de la secuencia,batch_size,Cantidad de componentes]
        sequence_length: Longitud de la secuencia
        batch_size: Tamaño de la muestra [Sugerible de 1]
        input_size: input size == 15 Componentes
        future: Cantidad de predicciones futuras o horas a predecir

        With iterator: True -> return output [Longitud de la secuencia + future,batch_size,Cantidad de componentes]
        With iterator: False -> return output , (h_t,c_t) [batch_size,Cantidad de componentes], [num_layers,batch_size,hidden_size], [num_layers,batch_size,hidden_size]
        """
        # Check if the size is [sequence_length,batch_size,input_size], so have len(x.size()) == 3
        
        if not isinstance(x, torch.Tensor):
            try:
                x = torch.from_numpy(x).float()
            except:
                assert False, "x must be a numpy array or a torch tensor of size [sequence_length,batch_size,input_size]"
        assert len(x.size()) == 3, "x must be a tensor of size [sequence_length,batch_size,input_size], but got size {}".format(x.size())
        # outputs
        if with_iteration == True:
            assert (future != 0), "future must be not 0 when with_iteration is True"
            outputs = []
            (h_t,c_t) = self.ini_hidden(1)
            #h_t = torch.zeros(self.num_layers,1,self.hidden_size)
            #c_t = torch.zeros(self.num_layers,1,self.hidden_size)
            
            for input_t in x.split(1,dim=0):
                out, (h_t,c_t) = self.lstm1(input_t,(h_t,c_t))
                out = self.LinearSeq(out)
                output = self.LOut(out)
                outputs.append(output)
            
            for i in range(future):
                out, (h_t,c_t) = self.lstm1(output, (h_t,c_t))
                out = self.LinearSeq(out)
                output = self.LOut(out)
                outputs.append(output)
            outputs = torch.cat(outputs,dim=0)
            return outputs
        else:
            assert future == 0, "future must be 0 when with_iteration is False"
            assert (h_t is not None and c_t is not None), "h_t and c_t must be not None"
            out, (h_t,c_t) = self.lstm1(x,(h_t,c_t))
            out = self.LinearSeq(out)
            output = self.LOut(out)
            return output, h_t,c_t

    def fit(self,x_train, y_train, epochs : int = 1, batch_size : int = 1, verbose : bool = True):
        """
        x_train: input [Longitud de la secuencia,batch_size,Cantidad de componentes]
        y_train: output [Longitud de la secuencia,batch_size,Cantidad de componentes]
        epochs: Cantidad de iteraciones
        batch_size: Tamaño de la muestra [Sugerible de 1]
        verbose: Mostrar el progreso de entrenamiento
        """
        assert batch_size == 1, "batch_size must be 1, we don't support batch training yet"
        try:
            from tqdm import tqdm
        except:
            assert False, "tqdm is not installed, please install it with 'pip install tqdm'"

        assert (x_train.size()[0] == y_train.size()[0]), "x_train and y_train must have the same size in the first dimension"
        assert (x_train.size()[1] == y_train.size()[1]), "x_train and y_train must have the same size in the second dimension"
        assert (x_train.size()[2] == self.input_size), "x_train must have the same size in the third dimension as the input_size"

        assert epochs > 0, "epochs must be greater than 0"
        assert batch_size > 0, "batch_size must be greater than 0"
        criterion = nn.MSELoss()
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        print ("Training...")
        for epoch in range(epochs):
            for i in tqdm(range(0, len(x_train), batch_size)):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                (h_t,c_t) = self.ini_hidden(batch_size)
                pred, h_t,c_t = self.forward(x_batch,h_t,c_t)
                loss = self.criterion(pred,y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if verbose:
                print ("Epoch: {} Loss: {}".format(epoch,loss.item()))


def Save_Outputs(outputs : Tensor, type: str= "csv", path :str= "outputs") -> None:
    """
    Guarda los outputs en un archivo con terminacion .type
    """
    if type == "csv":
        outputs = outputs.detach().numpy()
        np.savetxt(path+".csv",outputs,delimiter=",")
    elif type == "npy":
        outputs = outputs.detach().numpy()
        np.save(path+".npy",outputs)
    elif type == "json":
        try:
            import json
        except:
            assert False, "json is not installed, please install it with 'pip install json'"

        outputs = outputs.detach().numpy()
        with open(path+".json", "w") as outfile:
            json.dump(outputs.tolist(), outfile)
    else:
        assert False, "type must be csv, npy or json"