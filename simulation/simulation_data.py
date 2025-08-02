from scipy.stats import dirichlet
import torch
import numpy as np
class RealData:
    def __init__(self,alpha = None, lamda = None,sequence=None,num=None,J=None,L=None):
        self.sequence = sequence
        self.p = len(sequence)
        self.n = num
        self.J = J
        self.alpha = alpha
        self.lamda = lamda
        self.L = L

    def generate_parameters_r(self):
        element = [i for i in self.sequence]
        theta_0 = 1 + np.random.uniform(size = self.p)
        theta_0 = theta_0/theta_0.sum()
        Theta = dirichlet.rvs([self.alpha]*self.p, size = self.J)
        w = np.random.choice([0, 1], p = [1 - self.lamda, self.lamda], size = self.n)
        # generate a
        a = torch.randint(0,self.L - self.J + 1, size = (self.n, 1))
        # generate r
        R = []

        for i in range(self.n):
            temp = []
            # true sequence
            if w[i] ==1:
                for j in range(self.L):
                    if j < a[i][0] or j > a[i][-1]:
                        temp.append(np.random.choice(element, p = theta_0))
                    else:
                        temp.append(np.random.choice(element, p = Theta[j - a[i][0]]))
            # fake sequence
            else:
                temp = list(np.random.choice(element, p = theta_0, size = self.L))
            temp = "".join(temp)
            R.append(temp)
        self.R = R
        self.theta_0 = theta_0
        self.Theta = Theta
        return R,a,theta_0,Theta,w
    
    
    def generate_y(self):
        def encode(r:list,a:int) -> np.array:
            element = [i for i in "ACDEFGHIKLMNPQRSTVWY"]
            if type(a) == int:
                vector = np.zeros(len(element))
                index = element.index(r[a])
                vector[index] = 1
            else:
                vector = np.zeros((len(a), len(element)))
                index = [element.index(r[i]) for i in a]
                
                for i in range(len(index)):
                    vector[i][index[i]] = 1

            return np.array(vector)
        R,A,theta_0,Theta,w = self.generate_parameters_r()
        n, J = len(R), len(Theta)
        y = np.zeros(n)
        for i in range(n):
            r = R[i]
            a = A[i]
            vector = encode(r, a) 
            beta = np.random.randn(J,len(theta_0))
            y[i] = (vector * beta).sum(axis = 1).mean() + np.random.randn()
        #sigmoid
        y = list(map(lambda x: 1/(1 + np.exp(x)), y))
        self.y,self.R,self.A,self.theta_0,self.Theta,self.w = torch.tensor(y),R,A,theta_0,Theta,w
        return self.y,self.R,self.A,self.theta_0,self.Theta,self.w