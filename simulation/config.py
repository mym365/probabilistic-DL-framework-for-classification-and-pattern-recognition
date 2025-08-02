class SimConfig:
    """simulation params"""
    # data generation
    n = 500  # sample size
    J = 10   # motifs length
    L = 15   # sequence length
    p = 20   # motifs number
    elements = "ACDEFGHIKLMNPQRSTVWY" #Amino acid type set
    random_seed = 42
    
    # network params
    input_dim = 200
    hidden_dim1 = 60
    hidden_dim2 = 20
    lr = 0.05 
    iter_num = 200
