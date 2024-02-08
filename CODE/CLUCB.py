import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from numpy.linalg import multi_dot
from scipy.linalg import sqrtm
from scipy.sparse import csgraph
import os

N_TRIAL = 30000
N_ARMS = 100
N_FEATURE = 10
M = 29 #can change to different numbers of a monitoring capacity
K = 3
np.random.seed(123)


# In[3]:

#case1
Y_1 = np.genfromtxt(r'SET1/N100S1_Y_x10.csv',delimiter=',')
Beta = np.genfromtxt(r'SET1/N100S1_Beta_x10.csv',delimiter=',')

X_1 = {}
for i in range(N_ARMS):
    name = 'SET1/X10/N100/'+ 'N100S1_'+'X10_' + str(i+1) + '.csv'
    readX = np.genfromtxt(name,delimiter=',')
    X_1[i] = readX.T

#case2
Y_2 = np.genfromtxt(r'Y2_set1.csv',delimiter=',', skip_header=1)

# Initialize q and C
np.random.seed(3) #3 #59
vec_q = np.array([np.random.rand() for _ in range(K * N_FEATURE)])
# vec_q = q[~np.isnan(q)]
# vec_C: C (NK x 1)
np.random.seed(42)
longvec_C = np.array([np.random.rand() for _ in range(N_ARMS * K)])
matrix_c = longvec_C.reshape(N_ARMS, K)
vec_C = matrix_c

# Getting the optimal selection for all 30000 iterations

oracle_lst = []
true_choice = []
new_y = -1 * Y_1 + 30 # for case2 set new_y = Y_2
for t in np.arange(N_TRIAL):
	# Find indices of M highest arms
	all_reward_t = [new_y.T[t, arm] for arm in np.arange(N_ARMS)]
	chosen_arms = np.array(all_reward_t).argsort()[-M:][::-1]
	# Sum of M highest rewards
	oracle_payoff_t = np.sum([new_y.T[t, choice] for choice in chosen_arms])
	# Append to the list
	oracle_lst.append(oracle_payoff_t)
	true_choice.append(chosen_arms)
oracle_case1 = np.array(oracle_lst)

# for case2
# Generate a feature vector X_{it} (1, t, t^2) for 30000 iterations
# X_1_lst = []
# for T in np.arange(N_TRIAL):
#   X_1t_lst = []
#   for arm in np.arange(N_ARMS):
#     temp = []
#     temp.append(1)
#     temp.append(0.001*(T+1))
#     temp.append((0.001*(T+1))**2)
#     X_1t_lst.append(np.array(temp))
#     # np.append(X_1, X_1t)
#   X_1_lst.append(np.array(X_1t_lst))
# X_1 = np.array(X_1_lst)

def plot_regrets(results, oracle):
    [plt.plot(make_regret(payoff=x['r_payoff'], oracle=oracle), label="alpha: "+str(alpha)) for (alpha, x) in results.items()]

def make_regret(payoff, oracle):
    return np.cumsum(oracle - payoff)

def plot_l2_norm_diff(results, true_theta, n_trial, n_arms, n_feature, FInv_Init, true_choice, K):
    [plt.plot(make_l2_norm_diff(x['theta_lst'], x['q_lst'], x['C_lst'], true_theta, n_trial, n_arms, n_feature, FInv_Init, true_choice, K), label="alpha: "+str(alpha)) for (alpha, x) in results.items()]    
# l2_norm_diff
def make_l2_norm_diff(theta, q_lst, C_lst, true_theta, n_trial, n_arms, n_feature, FInv_Init, true_choice, K):
    diff = []
    for t in range(n_trial):
        theta_matrix = (q_lst[t].reshape(n_feature, K).T).dot((FInv_Init.dot(C_lst[t]).real).reshape(n_arms, K).T).T
        diff_t = 0
        for arm in range(n_arms):
            diff_vec = np.array([theta_matrix[arm] - (true_theta[arm])])
            diff_t += np.linalg.norm(diff_vec, ord=2) 
        diff.append(diff_t)
    return np.array(diff)

# X transformation from a sparse matrix
def X_reshape(X, X_tr, t, K, n_arms, n_feature):  #
  	for arm in range(1, n_arms):
    	X_tr = np.concatenate((X_tr,np.kron(np.identity(n = K),X[arm].reshape(-1,1))), axis = 1)
  	return X_tr

# convert to a sparse matrix
def X_to_X_m(X, t, arm_choice, n_arms, n_feature):
  	X_m = np.copy(X[t])
  	for arm in np.arange(n_arms): # N x p
    	if arm not in arm_choice:
      		X_m[arm] = np.zeros(shape=n_feature)
  	return X_m

# Create the F matrix
def constructAdjMatrix(W, n, threshold): #m
    Adj_mat = np.zeros(shape = (n, n))
    for ui in range(n):
        for uj in range(n):
            Adj_mat[ui][uj] = W[ui][uj]
        # trim the graph
            for i in range(n):
                if W[ui][i] <= threshold:
                    Adj_mat[ui][i] = 0;
    return Adj_mat
def constructLaplacianMatrix(W, n, Gepsilon):
    G = W.copy()
    #Convert adjacency matrix of weighted graph to adjacency matrix of unweighted graph
    for i in range(n):
        for j in range(n):
            if G[i][j] > 0:
                G[i][j] = 1
    L = csgraph.laplacian(G, normed = False)
    I = np.identity(n = G.shape[0])
    GW = I + Gepsilon*L  # W is a double stochastic matrix
    return GW.T

# Create the F_{\otimes}^{-1} matrix
lda = 0
T = 0.5
test_adj = constructAdjMatrix(W, N_ARMS, T)
test_F = constructLaplacianMatrix(test_adj, N_ARMS, lda)
FInv_Init = sqrtm(np.linalg.inv(np.kron(test_F, np.identity(n=N_FEATURE))))

#eta_1, eta_2, lambda = tuning param.
#alpha_q, alpha_c = exploration parameters for q and C
#X = feature matrix
#Y = true reward matrix
#F = the sum of the identity matrix (N x N) and the laplacian matrix E
#FInv = The inverse of the sqrt of the kronecker product between the F and the identity matrix (K x K)
#m = number of selected arms
#K = number of latent structures

def CoCoUCB(eta_1, eta_2, alpha_q, alpha_c, X, Y, init_q, init_c, m, K, FInv, X_to_X_m, X_reshape, oracle):
    print("CoCoUCB with eta_1:", eta_1, "eta_2:", eta_2, "alpha_q:", alpha_q, "alpha_c:", alpha_c)
    n_trial, n_arms, n_feature = X.shape
    #n_arms = N | n_trial = t | n_features = p | K

    # 1. Initialize objects
    # 1.1. Output objects                     
    arm_choice = np.empty(shape=(n_trial, m), dtype=int)
    r_payoff = np.empty(n_trial)   
    c_payoff = np.empty(n_trial) 
    p = np.empty(shape=(n_trial, n_arms)) 
    acc = np.empty(shape=(n_trial, n_arms))
    cum_regret = np.empty(n_trial)
    theta_lst = np.empty(shape=(n_trial, n_arms * n_feature))
    C_lst = np.empty(shape=(n_trial, n_arms * K))
    q_lst = np.empty(shape=(n_trial, K * n_feature))

    
    # 1.2. Intialize q and C_tilde
    q = np.empty(shape = (n_trial + 1, K * n_feature)) #Kp x 1
    C_tilde = np.empty(shape = (n_trial + 1, n_arms * K)) #NK x 1
    q[0] = init_q
    C_tilde[0] = init_c
    
    te_q = np.empty(shape = (n_trial + 1, K * n_feature)) #Kp x 1
    te_C_tilde = np.empty(shape = (n_trial + 1, n_arms * K)) #NK x 1
    
    # 1.4. A, b, D, d
    A = eta_1 * np.identity(n = K * n_feature) #Kp x Kp
    b = np.zeros(shape=K * n_feature)          #Kp x 1
    D = eta_2 * np.identity(n = n_arms * K)    #NK x NK
    d = np.zeros(shape=n_arms * K)             #NK x 1
    inv_A = np.linalg.inv(A)
    inv_D = np.linalg.inv(D)
    
    # 2. Algorithm
    for t in np.arange(n_trial):
        for a in np.arange(n_arms):
            
          #x_tr for q case 
          X_temp = X_to_X_m(X, t, [a], n_arms, n_feature)    
          X_tr_init = np.kron(np.identity(n = K),X_temp[0].reshape(-1,1))
          X_tr = X_reshape(X_temp, X_tr_init, t, K, n_arms, n_feature) #Kp x NK 
          
          #x_tr for c case
          X_tilde = FInv.dot(X_to_X_m(X, t, [a], n_arms, n_feature).flatten()) #Np x 1
          q_block = (block_diag(*[q[t].reshape((K, n_feature)) for _ in np.arange(n_arms)])).T #Np x NK  
        
          #Calculate cb_q and cb_c
          #cb_q  
          X_q_a = X_tr.dot(FInv.dot(C_tilde[t]))
          cb_q = alpha_q * np.sqrt(X_q_a.T.dot(inv_A).dot(X_q_a))
        
          #cb_c
          X_c = q_block.T.dot(X_tilde)
          cb_c = alpha_c * np.sqrt((X_c).T.dot(inv_D).dot(X_c))
        
          #Predictions
          p[t, a] = (FInv.dot(C_tilde[t]).T).dot(X_tr.T).dot(q[t]) + cb_q + cb_c
          acc[t, a]   = p[t, a] - Y[t, a]
        # Choose m best arms
        chosen_arms = p[t].argsort()[-m:][::-1]
        for i in np.arange(m):
          arm_choice[t][i] = chosen_arms[i]
        
        #Update A, b, D, d for each selected arm
        te_q[t] = q[t]
        te_C_tilde[t] = C_tilde[t]
        for chosen_arm in arm_choice[t]:
            X_tr_chosen_temp = X_to_X_m(X, t, [chosen_arm], n_arms, n_feature)
            X_tr_init_cs = np.kron(np.identity(n = K),X_tr_chosen_temp[0].reshape(-1,1))
            X_1_tr_chosen =  X_reshape(X_tr_chosen_temp, X_tr_init_cs, t, K, n_arms, n_feature)
            
            #x_tr for c case
            X_tilde_chosen = FInv.dot(X_to_X_m(X, t, [chosen_arm], n_arms, n_feature).flatten())
            # q_block_chosen = (block_diag(*[q[t].reshape((K, n_feature)) for _ in np.arange(n_arms)])).T
            q_block_chosen = (block_diag(*[te_q[t].reshape((K, n_feature)) for _ in np.arange(n_arms)])).T
            
            #Update  
            X_q = FInv.dot(te_C_tilde[t]).dot(X_1_tr_chosen.T)
            X_C_Tilde = q_block_chosen.T.dot(X_tilde_chosen)
            A = A + np.outer(X_q, X_q)
            b = b + Y[t, chosen_arm] * X_q
            D = D + np.outer(X_C_Tilde, X_C_Tilde)           
            d = d + Y[t, chosen_arm] * X_C_Tilde
        
        #inverse calculation
        inv_A = np.linalg.inv(A)
        inv_D = np.linalg.inv(D)
        te_q[t] = inv_A.dot(b)
        te_C_tilde[t] = inv_D.dot(d)
        q[t + 1] = te_q[t]
        C_tilde[t + 1] = te_C_tilde[t]
        #just for recording data
        q_lst[t] = q[t + 1]
        C_lst[t] = C_tilde[t + 1]
        theta_lst[t] = ((q[t + 1].reshape(n_feature, K)).dot(C_tilde[t + 1].reshape(K,n_arms)).flatten())
                
        # Cumulative rewards
        if t == 0:
            c_payoff[t] = np.sum([Y[t, choice] for choice in arm_choice[t]])
        else:
            c_payoff[t]   = c_payoff[t-1] + np.sum([Y[t, choice] for choice in arm_choice[t]])
        r_payoff[t]   = np.sum([Y[t, choice] for choice in arm_choice[t]])
        cum_regret[t] = np.sum(oracle[0:t+1] - r_payoff[0:t+1])
        if (t+1) % 1000 == 0:
            # print('TRIAL:',t,'DONE', '| arm selected:', chosen_arms)
            print('TRIAL:',t,'DONE', '| cum_regret:', cum_regret[t])
    return dict(theta_lst=theta_lst, q_lst=q_lst, C_lst=C_lst, q = q, C_tilde = C_tilde, p = p, arm_choice = arm_choice, r_payoff = r_payoff, c_payoff = c_payoff, acc=acc)

print('M:', M, 'lda:', lda, 'T:', T)
alpha_to_test = [0.5]
# for case2 set Y = Y_2.T
results_dict = {alpha: CoCoUCB(eta_1 = 0.3, eta_2 = 0.3, alpha_q = alpha, alpha_c = alpha, X=X_1, Y=(-1 * Y_1 + 30).T, init_q=vec_q, init_c=vec_C,m=M, K = K, FInv=FInv_Init, X_to_X_m=X_to_X_m, X_reshape=X_reshape, oracle=oracle_case1)\
                for alpha in alpha_to_test}
