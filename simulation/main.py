import torch
import argparse
from config import SimConfig
from simulation_data import RealData
from simulation import Model,Iter
def run_simulation():
    parser = argparse.ArgumentParser(description="simulation params")
    # 添加可调整的参数（默认值从配置文件取）
    parser.add_argument("--n", type=int, default=SimConfig.n)
    parser.add_argument("--lr", type=float, default=SimConfig.lr)
    parser.add_argument("--J", type=int, default=SimConfig.J)
    parser.add_argument("--L", type=int, default=SimConfig.L)
    parser.add_argument("--p", type=int, default=SimConfig.p)
    parser.add_argument("--elements", type=str, default=SimConfig.elements)
    parser.add_argument("--iter_num", type=int, default=SimConfig.iter_num)
    parser.add_argument("--input_dim", type=int, default=SimConfig.input_dim)
    parser.add_argument("--hidden_dim1", type=int, default=SimConfig.hidden_dim1)
    parser.add_argument("--hidden_dim2", type=int, default=SimConfig.hidden_dim2)
    args = parser.parse_args()
    
    print(f"样本量: {args.n}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.iter_num}")
    # basic params
    n,J,p,L,elements = args.n,args.J,args.p,args.L,args.elements
    # nn params
    element_to_index = {char: idx for idx, char in enumerate(elements)}
    def one_hot_encoding(seq):
        one_hot = torch.zeros((len(seq), len(elements)))
        for i, char in enumerate(seq):
            one_hot[i, element_to_index[char]] = 1
        return one_hot
    alpha, lamda = [0.05, 0.1, 0.15], [0.3, 0.5, 0.7]
    for alpha_i in alpha:
        for lamda_i in lamda:
            print("alpha = {}, lamda = {}".format(alpha_i, lamda_i))
            real_data = RealData(alpha = alpha_i, lamda = lamda_i,sequence=elements,num=n,J=J,L=args.L)
            y,R,true_a,true_theta_0,true_Theta,true_w= real_data.generate_y()
            for i in range(len(R)):
                R[i] = one_hot_encoding(R[i])
            L = torch.tensor([len(r) for r in R])
            for i in range(1,21):
                model = Model(R,y,L,J,p,args.input_dim,args.hidden_dim1,args.hidden_dim2)
                # get simulation results
                history_Q, Theta, theta_0,a,w,predict_score = Iter(model,iter_num=args.iter_num)


if __name__ == "__main__":
    run_simulation()