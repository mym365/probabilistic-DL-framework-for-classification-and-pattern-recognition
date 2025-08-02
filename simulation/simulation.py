import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
device = torch.device("cpu")


    
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
    def __init__(self,R,y,L,J,p,input_dim, hidden_dim1, hidden_dim2):
        '''
        params: n,J,p,L
        '''
        self.L = L
        self.device = torch.device("cpu")#("cuda" if torch.cuda.is_available() else "cpu")
        self.R = R
        self.y = y
        self.J = J
        self.p = p
        self.n = len(y)
        self.model = NetWork(input_dim, hidden_dim1, hidden_dim2).to(self.device).double()
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.05)
        #self.batch_size = batch_size



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
        self.sigma2, self.nn_loss = sigma2, sigma2
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


    def calculate_expectation(self,a,w,Theta,theta_0):
        n = len(a)
        nn_loss = 0
        for i in range(n):
            nn_loss += self.phi_i(i,a[i])
        self.nn_loss = nn_loss
        f_Theta = torch.where(Theta == 0, torch.tensor(0), torch.log(Theta))
        f_theta_0 = torch.where(theta_0 == 0, torch.tensor(0), torch.log(theta_0))

        xi_values = torch.zeros((n, self.J, self.p))
        zeta_values = torch.zeros((n, self.p))

        # value of xi and zeta 
        for i in range(n):
            xi_values[i] = self.xi(a[i], w[i], i)
            zeta_values[i] = self.zeta(a[i], w[i], i).sum(dim=0)

        temp1 = (xi_values * f_Theta).sum(dim=(1,2))
        temp2 = (zeta_values * f_theta_0).sum(dim=1)

        ##############
        temp = temp1.to(self.device) + temp2.to(self.device) - 0.5 *  torch.log(self.sigma2)
        Q = temp.sum()- self.nn_loss / (2 * self.sigma2)
        return Q

    


    #update a
    def sample_p_a(self, Theta, theta_0):
        a = torch.zeros((self.n,),dtype=torch.int64)
        for i in range(len(self.L)):
            pro = torch.zeros(self.L[i] - self.J + 1)
            for j in range(len(pro)):
                a0 = j
                loss = self.phi_i(i, a0)
                normal = torch.exp(-1/2 * loss / self.sigma2)
                temp = (Theta * self.h(self.R[i], a0)).sum(dim = 1).prod()  #1e10 for avoiding zero division
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
            pw1 = (Theta * self.h(self.R[i], a[i])).sum(dim = 1).prod()
            pw1 = (theta_0 * self.zeta(a[i], 1, i)).sum(dim = 1).prod() * pw1
            pw = torch.tensor([pw0,pw1])
            pro[i] = pw/pw.sum()
            w[i] = torch.distributions.Categorical(pro[i]).sample()
        return pro, w

    def shift_a(self, a, direction):
        if direction == -1:
            a -= 1
            a[a<0]  += 1
        elif direction == 1:
            a += 1
            a[a+self.J>self.L] -= 1
        return a

    def update_beta(self):
        self.sigma2 = self.nn_loss
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

def Iter(instance, iter_num = 200, epsilon = 0.0001):
    history_Q = []
    print("device:",instance.device)
    print("optimizer:",instance.optimizer)
    print("maximum iterations num:",iter_num)
    Theta, theta_0, w, a, sigma2 = instance.initilize_params()
    for i in range(1, iter_num+1):
        instance.optimizer.zero_grad()
        Q = instance.calculate_expectation(a,w,Theta,theta_0)
        #update parameters
        Theta, theta_0 = instance.update_params(a, w)
        a = instance.sample_p_a(Theta, theta_0)
        pro,w = instance.sample_p_w(a, Theta, theta_0)
        if i>50 and i%10 == 0:
            direction = random.choice([-1,1])
            temp_a = instance.shift_a(a, direction)
            temp_Theta, temp_theta_0 = instance.update_params(temp_a, w)
            _,temp_w = instance.sample_p_w(temp_a, temp_Theta, temp_theta_0)
            instance.optimizer.zero_grad()
            temp_Q = instance.calculate_expectation(temp_a,temp_w,temp_Theta,temp_theta_0)
            if temp_Q > Q:
                a,Theta,theta_0,w,Q = temp_a,temp_Theta,temp_theta_0,temp_w,temp_Q
                print("shift {} at iteration {}".format(direction, i))
        history_Q.append(Q)
        instance.update_beta()
        if len(history_Q) > 1 and abs(history_Q[-1] - history_Q[-2]) < epsilon:
            break
    predict_score = []
    for j in range(len(a)):
        input_data = instance.h(instance.R[j], a[j])
        true_y = instance.y[j].unsqueeze(0).to(instance.device).double()
        #print(input_data.shape, true_y.shape)
        flat_data = input_data.view(-1, instance.J * instance.p).to(instance.device)
        output = instance.model(flat_data.double()).squeeze(1)
        predict_score.append(output)

    return history_Q, Theta, theta_0,a,w,predict_score
