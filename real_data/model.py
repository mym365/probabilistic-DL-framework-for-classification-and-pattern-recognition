from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class NetWork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super().__init__()
        # input = (batch_size, J * p)
        self.input_dim = input_dim
        torch.set_default_dtype(torch.float64)
        self.linear = nn.Sequential(OrderedDict([
            ("input",nn.Linear(input_dim, hidden_dim1, dtype=torch.float64)),
            ("relu1",nn.ReLU()),

            ("linear1",nn.Linear(hidden_dim1, hidden_dim2, dtype=torch.float64)),
            ("relu2",nn.ReLU()),
            
            ("linear2",nn.Linear(hidden_dim2, 1, dtype=torch.float64)),
            ("sigmoid",nn.Sigmoid())
        ]))
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.linear(x)
        return x
    
class Model:
    def __init__(self,R,y,L,J,p,input_dim, hidden_dim1, hidden_dim2,lr):
        '''
        params: n,J,p,L
        '''
        self.L = torch.tensor(L)
        self.device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.R = R
        self.y = y
        self.J = J
        self.p = p
        self.model = NetWork(input_dim, hidden_dim1, hidden_dim2).to(self.device).double()
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)



    # initilize Theta, theta_0, w, a, sigma2
    def initilize_params(self):
        n = len(self.y)
        sigma2 = torch.tensor(1.0)
        Theta = torch.rand(size = (self.J,self.p))
        theta_0 = torch.rand(size = (self.p,))
        Theta, theta_0= Theta/Theta.sum(axis = 1, keepdim=True), theta_0/theta_0.sum()
        w = torch.bernoulli(torch.tensor([0.7]*n))
        #flat distribution
        a = torch.tensor([torch.randint(0, 1 + len(self.R[i]) - self.J, size=(1,)) for i in range(n)])
        self.sigma2, self.nn_loss = sigma2, sigma2*n
        return Theta, theta_0, w, a, sigma2

    # one-hot function
    def h(self,r,a):
        # r = encoder(r)
        input = r[a:a+self.J]
        return input

        
    # calculate loss
    def phi_i(self,index = None, a0 = None):
        input_data = self.h(self.R[index], a0)
        true_y = self.y[index].unsqueeze(0).to(self.device).double()
        #print(input_data.shape, true_y.shape)
        flat_data = input_data.view(-1, self.J * self.p).to(self.device)
        output = self.model(flat_data.double()).squeeze(1)
        loss = self.criterion(output, true_y)
        return loss


    # index = i
    def xi(self,a,w,index):
        if w == 0:
            return 1e-8 * torch.ones((self.J,self.p))
        else:
            return self.h(self.R[index], a) + 1e-8

    # index = i
    def zeta(self,a,w,index):
        all = self.R[index].double()
        if w == 0:
            return all + 1e-8
        else:
            return torch.cat((self.R[index][0:a], self.R[index][a+self.J:]), dim=0) + 1e-8

    def calculate_nnloss(self,a):
        n = len(a)
        nn_loss = 0
        for i in range(n):
            nn_loss += self.phi_i(i,a[i])
        return nn_loss
    def calculate_expectation(self,a,w,Theta,theta_0):
        n = len(a)
        f_Theta = torch.where(Theta == 0, torch.tensor(0), torch.log(Theta))
        f_theta_0 = torch.where(theta_0 == 0, torch.tensor(0), torch.log(theta_0))

        xi_values = torch.zeros((n, self.J, self.p))
        zeta_values = torch.zeros((n, self.p))

        for i in range(n):
            xi_values[i] = self.xi(a[i], w[i], i)
            zeta_values[i] = self.zeta(a[i], w[i], i).sum(dim = 0)

        temp1 = (xi_values * f_Theta).sum(dim=(1,2))
        temp2 = (zeta_values * f_theta_0).sum(dim=1)

        ##############
        temp = temp1.to(self.device) + temp2.to(self.device) - 0.5 *  torch.log(self.sigma2)
        Q = temp.sum()- self.nn_loss / (2 * self.sigma2)
        return Q


    #update a
    def sample_p_a(self, Theta, theta_0):
        a = torch.zeros((len(self.L),),dtype=torch.int64)
        for i in range(len(self.L)):
            pro = torch.zeros(self.L[i] - self.J + 1)
            for j in range(len(pro)):
                a0 = j
                loss = self.phi_i(i, a0)
                normal = torch.exp(-1/2 * loss / self.sigma2)
                temp = (Theta * self.h(self.R[i], a0)).sum(axis = 1).prod()  #1e10 for avoiding zero division
                temp *= (theta_0 * self.zeta(a0, 1, i)).sum(dim = 1).prod()
                temp += (theta_0 * self.zeta(a0, 0, i)).sum(dim = 1).prod()
                temp = temp.to(self.device)
                temp *= normal
                pro[j] = temp
             # ,check nan
            pro /= pro.sum()
            a[i] = torch.distributions.Categorical(pro).sample()
        return a

    #update w
    def sample_p_w(self,a,Theta,theta_0):
        n = len(a)
        pro = torch.zeros((n,2))
        w = torch.randint(0, 2, (n,))
        for i in range(n):
            pw0 = (theta_0 * self.zeta(a[i], 0, i)).sum(dim = 1).prod()
            pw1 = (Theta * self.h(self.R[i], a[i])).sum(axis = 1).prod()
            pw1 = (theta_0 * self.zeta(a[i], 1, i)).sum(dim = 1).prod() * pw1
            pw = torch.tensor([pw0,pw1])
            pro[i] = pw/pw.sum()
            w[i] = torch.distributions.Categorical(pro[i]).sample()
        return pro, w

    def shift_a(self, a, direction):
        a = a.clone()
        if direction == -1:
            a -= 1
            a[a<0]  += 1
        elif direction == 1:
            a += 1
            a[a+self.J>self.L] -= 1
        return a

    def update_beta(self):
        self.nn_loss.backward()
        self.optimizer.step()
        
    def update_params(self,a,w):
        #update w
        n = len(a)
        Theta = torch.zeros((self.J,self.p))
        theta_0 = torch.zeros(self.p)
        #update Theta, theta_0
        xi, zeta = 0, 0
        for i in range(n):
            xi += self.xi(a[i],w[i],i)
            zeta += self.zeta(a[i],w[i],i).sum(dim = 0)
        Theta = xi/xi.sum(dim = 1, keepdim=True)
        theta_0 = zeta/zeta.sum()
        return Theta, theta_0

def model_fit_predict(tools,train_R,test_R,test_Y,L,iter_num):
    likelihood = []
    Theta, theta_0, w, a, sigma2 = tools.initilize_params()
    for i in tqdm(range(1,iter_num+1)):
        tools.sigma2 = tools.nn_loss/len(a)
        a = tools.sample_p_a(Theta,theta_0)
        _, w = tools.sample_p_w(a,Theta,theta_0)
        Theta, theta_0 = tools.update_params(a,w)
        tools.optimizer.zero_grad()
        tools.nn_loss = tools.calculate_nnloss(a)
        tools.update_beta()
        Q = tools.calculate_expectation(a,w,Theta,theta_0)
        #shift mode
        if i>50 and i< 200 and i%10 == 0:
            info = -(Theta * np.log(Theta)).sum(axis = 1)
            info = info/info.sum()
            direction = -1 if info[0] < info[-1] else 1

            temp_a = tools.shift_a(a,direction)
            _, temp_w = tools.sample_p_w(temp_a,Theta,theta_0)
            temp_Theta, temp_theta_0 = tools.update_params(temp_a,temp_w)
            tools.optimizer.zero_grad()
            before_loss = tools.nn_loss
            tools.nn_loss = tools.calculate_nnloss(a)
            tools.update_beta()
            temp_Q = tools.calculate_expectation(temp_a,temp_w,temp_Theta,temp_theta_0)
            if temp_Q > Q:
                print('shift:f{direction}')
                a = temp_a
                Theta, theta_0 = temp_Theta, temp_theta_0
                Q = temp_Q
            else:
                tools.nn_loss = before_loss
        likelihood.append(Q)
    tools.model.eval()
    L = torch.tensor([len(r) for r in test_R])
    tools.R, tools.y, tools.L = test_R, test_Y, L
    test_a = tools.sample_p_a(Theta, theta_0)
    predict_y = []
    for i in range(len(test_a)):
        input_data = tools.h(tools.R[i], test_a[i])
        flat_data = input_data.view(-1, tools.J * tools.p).to(tools.device)
        predict_y.append(tools.model(flat_data.double()).squeeze(1))
    return likelihood, Theta, theta_0, w, a, predict_y