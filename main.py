import torch
import scipy.io as sio
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torch.utils.data as Data
import numpy as np
from train_loop import train_loop
from test_loop import test_loop
import torch.nn.functional as F
from obtain_nn_input_output_adaptive import obtain_nn_input_output_adaptive


#import tx and txData
index=0
N=10000
for i in range(30,71,20):
    for j in range(140,170,10):
        for k in range(65,86,10):
            pathtx = 'data_for_training/tx_' + str(i) + '_'+str(j)+'_'+str(k)+'.mat'
            pathtxData = 'data_for_training/txData_' + str(i) + '_'+str(j)+'_'+str(k)+'.mat'

            data_tx = 'data_tx_' + str(i) + '_' + str(j) + '_' + str(k)
            data_txData = 'data_txData_' + str(i) + '_' + str(j) + '_' + str(k)

            loadtx = data_tx + '=' + 'sio.loadmat(\'' + pathtx + '\')'
            loadtxData = data_txData + '=' + 'sio.loadmat(\'' + pathtxData + '\')'

            exec(loadtx)
            exec(loadtxData)

            data_tx_r = data_tx + '_r' + '=' + data_tx + '.get(\'tx\').real'
            data_tx_i = data_tx + '_i' + '=' + data_tx + '.get(\'tx\').imag'

            data_txData_r = data_txData + '_r' + '=' + data_txData + '.get(\'txData\').real'
            data_txData_i = data_txData + '_i' + '=' + data_txData + '.get(\'txData\').imag'

            exec(data_tx_r)
            exec(data_tx_i)

            exec(data_txData_r)
            exec(data_txData_i)

            exec(data_tx + '_r' + '=' + data_tx + '_r' + '[0:N]')
            exec(data_tx + '_i' + '=' + data_tx + '_i' + '[0:N]')
            exec(data_txData + '_r' + '=' + data_txData + '_r' + '[0:N]')
            exec(data_txData + '_i' + '=' + data_txData + '_i' + '[0:N]')



#import PAPR data
pathpapr = 'data_for_training/PAPR_cal.mat'
data_papr = 'data_papr'
loadpapr = data_papr + '=' + 'sio.loadmat(\'' + pathpapr + '\')'
exec(loadpapr)
PAPR=data_papr.get('PAPR_cal')

#memory depth of the proposed model
M=7

#obtain NN's input and output
for i in range(30,71,20):
    for j in range(140,170,10):
        for k in range(65,86,10):
            var1 = 'data_tx_'+str(i)+'_'+str(j)+'_'+str(k)+'_r'+'[0:N]'
            var2 = 'data_tx_'+str(i)+'_'+str(j)+'_'+str(k)+'_i'+'[0:N]'
            var3 = 'data_txData_'+str(i)+'_'+str(j)+'_'+str(k)+'_r'+'[0:N]'
            var4 = 'data_txData_'+str(i)+'_'+str(j)+'_'+str(k)+'_i'+'[0:N]'
            var5 = 'M'

#trasmission BW
            if i == 30:
                fd = 28.08
            if i == 40:
                fd = 38.16
            if i == 50:
                fd = 47.88
            if i == 60:
                fd = 58.32
            if i == 70:
                fd = 68.04
            if i == 80:
                fd = 78.12
            if i == 90:
                fd = 88.20
            if i == 100:
                fd = 98.28

#operating conditions
            var6_1 = fd / 68.07
            var6_2 = np.array([np.power(10,-j/100)]) / np.power(10,-140/100)
            var6_3 = PAPR[index] / 8.5
            HYPER = np.vstack([var6_1, var6_2, var6_3])

            exec('[nnin_'+str(i)+'_'+str(j)+'_'+str(k)+','+'nnout_'+str(i)+'_'+str(j)+'_'+str(k)+']=obtain_nn_input_output_adaptive('+var1+','+var2+','+var3+','+var4+','+var5+','+'HYPER'+')')
            index = index + 1

temp_in = nnin_30_140_65
temp_out = nnout_30_140_65

index=1
for i in range(30,71,20):
    for j in range(140,170,10):
        for k in range(65,86,10):
            if i==30 and j==140 and k==65:
                continue
            else:
                exec('temp_in=np.vstack([temp_in'+','+'nnin_'+str(i)+'_'+str(j)+'_'+str(k)+'])')
                exec('temp_out=np.vstack([temp_out'+','+'nnout_'+str(i)+'_'+str(j)+'_'+str(k)+'])')


train_dataset = Data.TensorDataset(torch.from_numpy(temp_in).float(), torch.from_numpy(temp_out).float())

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#define the proposed model
class PROPOSED(nn.Module):
    def __init__(self):
        super(PROPOSED, self).__init__()
        self.fc1 = nn.Linear(16, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 12)
        self.fc5 = nn.Linear(12, 2)
        self.fcgating1 = nn.Linear(3, 20) #relu gating
        self.fcgating2 = nn.Linear(20, 20)#relu gating
        self.fcgating3 = nn.Linear(20, 12)  # relu gating
    def forward(self, x):
        # gating
        gating = self.fcgating1(x[:,16:19])
        gating = torch.relu(gating)
        gating = self.fcgating2(gating)
        gating = torch.relu(gating)
        gating = self.fcgating3(gating)
        gating = torch.sigmoid(gating)

        # BOTDNN
        fc1_out = self.fc1(x[:,0:16])

        for i in range(10):
            VDM_INPUT = fc1_out[:,i*2:i*2+2]
            AMPLITUDE = VDM_INPUT**2
            AMPLITUDE = torch.sqrt(AMPLITUDE.sum(dim=1))
            temp_amplitude = torch.stack([AMPLITUDE, AMPLITUDE ** 2], dim=1)
            AMPLITUDE = torch.stack([AMPLITUDE,AMPLITUDE],dim=1)
            temp_sin_cos =  torch.div(VDM_INPUT,AMPLITUDE)
            if i==0:
                output_AMPLITUDE = temp_amplitude
                output_SIN_COS =temp_sin_cos
            else:
                output_AMPLITUDE = torch.cat([output_AMPLITUDE,temp_amplitude],dim=1)
                output_SIN_COS = torch.cat([output_SIN_COS, temp_sin_cos], dim=1)

        fc2_out = self.fc2(output_AMPLITUDE)
        #gating
        fc2_out = torch.tanh(fc2_out)

        X_stack = torch.cat([fc2_out, fc2_out], dim=1)
        recov_out = X_stack * output_SIN_COS
        fc3_out = self.fc3(recov_out)

        fc4_out = self.fc4(fc3_out)
        fc4_out = F.tanh(fc4_out)

        fc4_out = fc4_out * gating

        out = self.fc5(fc4_out)

        return out



model = PROPOSED()
learning_rate = 1e-3
batch_size = 64
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#3000 epochs are required
epochs = 3000
NMSE_CHECK=100

#training
for t in range(epochs):
    if t % 20==0:
        learning_rate=learning_rate*0.95
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    curr_NMSE = test_loop(train_dataloader, model)
    print("Train NMSE")
    print(curr_NMSE)

#obtain the DPD signal and save
    if t > 1000 and curr_NMSE < NMSE_CHECK:
        NMSE_CHECK = curr_NMSE
        pathpapr = 'data_for_training/PAPR_ver.mat'
        data_papr = 'data_papr'
        loadpapr = data_papr + '=' + 'sio.loadmat(\'' + pathpapr + '\')'
        exec(loadpapr)
        PAPR = data_papr.get('PAPR_ver')

        index = 0
        for i in range(30, 71, 10):
            for j in range(140, 161, 5):
                for k in range(65, 86, 5):
                    pathtx = 'data_for_training/tx_' + str(i) + '_' + str(j) + '_' + str(
                        k) + '.mat'
                    data_tx = sio.loadmat(pathtx)

                    data_tx_r = data_tx.get('tx').real
                    data_tx_i = data_tx.get('tx').imag
                    if i == 30:
                        fd = 28.08
                    if i == 40:
                        fd = 38.16
                    if i == 50:
                        fd = 47.88
                    if i == 60:
                        fd = 58.32
                    if i == 70:
                        fd = 68.04
                    if i == 80:
                        fd = 78.12
                    if i == 90:
                        fd = 88.20
                    if i == 100:
                        fd = 98.28

                    var6_1 = fd / 68.07
                    var6_2 = np.array([np.power(10, -j / 100)]) / np.power(10, -140 / 100)
                    var6_3 = PAPR[index] / 8.5
                    HYPER = np.vstack([var6_1, var6_2, var6_3])

                    [temp_in, temp] = obtain_nn_input_output_adaptive(data_tx_r, data_tx_i, data_tx_r, data_tx_i, M,
                                                                      HYPER)
                    test_dataset = Data.TensorDataset(torch.from_numpy(temp_in).float(), torch.from_numpy(temp).float())
                    test_dataloader = DataLoader(test_dataset, batch_size=200000, shuffle=False)

                    with torch.no_grad():
                        for X, y in test_dataloader:
                            ILC = model(X)

                    ILC = np.array(ILC)
                    path_save = 'DPD_signal/' + 'ILC_' + str(i) + '_' + str(
                        j) + '_' + str(k) + '.mat'

                    sio.savemat(path_save, {'ILC': ILC})
                    index = index + 1





