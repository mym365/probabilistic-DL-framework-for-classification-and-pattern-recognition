class Config:
    J = 9   # motifs length
    p = 20   # motifs number
    elements = "ACDEFGHIKLMNPQRSTVWY" #Amino acid type set
    geno_num = 4
    random_seed = 42
    
    # network params
    input_dim = 180
    hidden_dim1 = 60
    hidden_dim2 = 20
    lr = 0.05 
    iter_num = 300
