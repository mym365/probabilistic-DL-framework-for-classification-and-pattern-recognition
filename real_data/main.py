import argparse
from config import Config
from model import Model, model_fit_predict
from data import get_train_test_data
def real_data():
    parser = argparse.ArgumentParser(description="simulation params")
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--J", type=int, default=Config.J)
    parser.add_argument("--p", type=int, default=Config.p)
    parser.add_argument("--elements", type=str, default=Config.elements)
    parser.add_argument("--iter_num", type=int, default=Config.iter_num)
    parser.add_argument("--input_dim", type=int, default=Config.input_dim)
    parser.add_argument("--hidden_dim1", type=int, default=Config.hidden_dim1)
    parser.add_argument("--hidden_dim2", type=int, default=Config.hidden_dim2)
    parser.add_argument("--geno_num", type=int, default=Config.geno_num)
    args = parser.parse_args()

    for j in range(args.geno_num):
        train_R,train_Y,test_R,test_Y = get_train_test_data(j)
        L = [len(r) for r in train_R]
        real_data_model = Model(train_R, train_Y,L,args.J,args.p,args.input_dim, args.hidden_dim1, args.hidden_dim2,args.lr)
        # get result of real data
        likelihood, Theta, theta_0, w, a, predict_y= model_fit_predict(real_data_model,train_R,test_R,test_Y, L,iter_num=args.iter_num)

if __name__ == "__main__":
    real_data()