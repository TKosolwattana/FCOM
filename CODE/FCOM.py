#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import block_diag
from numpy.linalg import multi_dot
from scipy.linalg import sqrtm
from scipy.sparse import csgraph
from scipy import linalg
import os


# In[2]:


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


def make_regret(payoff, oracle):
    return np.cumsum(oracle - payoff)

def plot_regrets(results, oracle):
    [plt.plot(make_regret(payoff=x['r_payoff'], oracle=oracle), label="alpha: "+str(alpha)) for (alpha, x) in results.items()]


# In[7]:


# X transformation to a sparse matrix
def X_reshape(X, X_tr, t, K, n_arms, n_feature):  #
  for arm in range(1, n_arms):
    X_tr = np.concatenate((X_tr,np.kron(np.identity(n = K),X[arm].reshape(-1,1))), axis = 1)
  return X_tr

# convert to a sparse matrix -> convert to a long sparse vector with flatten() -> Np x 1
def X_to_X_m(X, t, arm_choice, n_arms, n_feature): 
  X_m = np.copy(X[t])
  for arm in np.arange(n_arms): # N x p
    if arm not in arm_choice:
      X_m[arm] = np.zeros(shape=n_feature)
  return X_m


# In[8]:


def upload(gammaU, IDclient, A_loc, A_up_buff):    
    return (linalg.det(A_loc[IDclient])) > gammaU*linalg.det(A_loc[IDclient] - A_up_buff[IDclient])


# In[9]:


def Fed_CLUCB(N_TRIAL, N_ARMS, N_FEATURE, break_point, eta_1, eta_2, alpha_q, alpha_c, X, Y, init_q, init_c, m, K, X_to_X_m, X_reshape, oracle, gammaU, iterations):
    print('gammaU:', gammaU, alpha_q, alpha_c, iterations)
    # n_trial, n_clients, n_feature = X.shape
    n_trial = N_TRIAL
    n_clients = N_ARMS
    n_feature = N_FEATURE
    # 1.1. Output objects
    pass_comm = 0
    totalCommCost = 0
    client_choice = np.empty(shape=(n_trial, m), dtype=int)
    r_payoff = np.empty(n_trial)   
    c_payoff = np.empty(n_trial) 
    cum_regret = np.empty(n_trial)
    p = np.empty(shape=(n_trial, n_clients))
    cum_totalCommCost = np.empty(n_trial)
    
    # 1.2. Intialize local statistics
    A_loc = np.array([eta_1 * np.identity(n=K * n_feature) for _ in np.arange(n_clients)])
    A_up_buff = np.array([np.zeros((K * n_feature, K * n_feature)) for _ in np.arange(n_clients)])
    b_loc = np.array([np.zeros(shape=K * n_feature)  for _ in np.arange(n_clients)])
    b_up_buff = np.array([np.zeros(shape=K * n_feature)  for _ in np.arange(n_clients)])
    q_loc = np.empty(shape = (n_trial, n_clients, K * n_feature)) #Kp x 1
    A_down_buff = np.array([np.zeros((K * n_feature, K * n_feature)) for _ in np.arange(n_clients)])
    b_down_buff = np.array([np.zeros(shape=K * n_feature)  for _ in np.arange(n_clients)])
    
    D_loc = np.array([eta_2 * np.identity(n= K) for _ in np.arange(n_clients)])
    d_loc = np.array([np.zeros(shape= K)  for _ in np.arange(n_clients)])
    c_loc = np.empty(shape = (n_trial, n_clients, K)) #K x 1 (n clients)
    
    # temp parameters
    te_q_loc = np.empty(shape = (n_trial, n_clients, K * n_feature)) #Kp x 1
    te_c_loc = np.empty(shape = (n_trial, n_clients, K)) #K x 1 (n clients)
    
    #add initialization for each client
    for b in np.arange(n_clients): 
        q_loc[0, b] = init_q
        c_loc[0, b] = init_c[b]
        te_q_loc[0, b] = init_q
        te_c_loc[0, b] = init_c[b]
        
    # 1.3 Global statistics
    A_gob = eta_1 * np.identity(n=K * n_feature)  
    b_gob = np.zeros(shape=K * n_feature)     
    q_gob = np.zeros(shape=K * n_feature)
    
    # 2. Algorithm
    for t in np.arange(n_trial):
        for a in np.arange(n_clients):
            #Calculate inv(A_loc[a]), inv(D_loc[a]), q_loc[t,a], c_loc[t,a]
            inv_A = np.linalg.inv(A_loc[a])
            inv_D = np.linalg.inv(D_loc[a])
            if t != 0:
                q_loc[t, a] = inv_A.dot(b_loc[a])
                c_loc[t, a] = inv_D.dot(d_loc[a])
                te_q_loc[t, a] = q_loc[t, a]
                te_c_loc[t, a] = c_loc[t, a]
        
            #X Transformation 
            X_tr = np.kron(np.eye(K), X[a][t].T)
            
            #Calculate cb_q and cb_c
            #cb_q  
            X_q_a = c_loc[t, a].dot(X_tr)
            cb_q = alpha_q * np.sqrt(X_q_a.dot(inv_A).dot(X_q_a.T))
            
            #cb_c
            X_c = X_tr.dot(q_loc[t, a])
            cb_c = alpha_c * np.sqrt((X_c).T.dot(inv_D).dot(X_c))
            
            #Predictions
            p[t, a] = c_loc[t, a].dot(X_tr).dot(q_loc[t, a]) + cb_q + cb_c #FInv.dot
            
        chosen_clients = p[t].argsort()[-m:][::-1]
        for i in np.arange(m):
            client_choice[t][i] = chosen_clients[i]
        
        
        # each client solve for q and c iteratively and locally using ALS
        
        for chosen_client in client_choice[t]:
            for j in np.arange(iterations):
                X_tr_chosen = np.kron(np.eye(K), X[chosen_client][t].T)
                X_q = (te_c_loc[t, chosen_client].dot(X_tr_chosen)).T
                X_C_Tilde = X_tr.dot(te_q_loc[t, chosen_client])
                
                # client local buffers update
                A_up_buff[chosen_client] = A_up_buff[chosen_client] + np.outer(X_q, X_q) 
                b_up_buff[chosen_client] = b_up_buff[chosen_client] + Y[t, chosen_client] * X_q
                
                A_loc[chosen_client] = A_loc[chosen_client] + np.outer(X_q, X_q)
                b_loc[chosen_client] = b_loc[chosen_client] + Y[t, chosen_client] * X_q
                D_loc[chosen_client] = D_loc[chosen_client] + np.outer(X_C_Tilde, X_C_Tilde)           
                d_loc[chosen_client] = d_loc[chosen_client] + Y[t, chosen_client] * X_C_Tilde
                
                te_q_loc[t, chosen_client] = np.linalg.inv(A_loc[chosen_client]).dot(b_loc[chosen_client])
                te_c_loc[t, chosen_client] = np.linalg.inv(D_loc[chosen_client]).dot(d_loc[chosen_client])
                
            
        #each client check upload conditions whether to upload to the server
        # for chosen_client in client_choice[t]:
            c_loc[t, chosen_client] = te_c_loc[t, chosen_client]
            
            if upload(gammaU, chosen_client, A_loc, A_up_buff):
                totalCommCost += 1
                pass_comm += 1
                # update server's statistics
                A_gob += A_up_buff[chosen_client] 
                b_gob += b_up_buff[chosen_client]
                
                # update server's download buffer for other clients
                for clientID in np.arange(n_clients):
                    if clientID != chosen_client:
                        A_down_buff[clientID] += A_up_buff[chosen_client]
                        b_down_buff[clientID] += b_up_buff[chosen_client]
                        
                # clear client's upload buffer
                A_up_buff[chosen_client] = np.zeros((K * n_feature, K * n_feature))
                b_up_buff[chosen_client] = np.zeros(shape=K * n_feature)
                
                q_gob = np.linalg.inv(A_gob).dot(b_gob)          
        
        # Send all statistics back to all clients
        if total_comm > 0:
            for cli in np.arange(n_clients):
                totalCommCost += 1
                total_comm = 0 
                A_loc[cli] += A_down_buff[cli]
                b_loc[cli] += b_down_buff[cli]
                    
                # clear cserver's download buffer
                A_down_buff[cli] = np.zeros((K * n_feature, K * n_feature))
                b_down_buff[cli] = np.zeros(shape=K * n_feature)

                
        # Cumulative regret
        r_payoff[t] = np.sum([Y[t, choice] for choice in client_choice[t]])      
        cum_regret[t] = np.sum(oracle[0:t+1] - r_payoff[0:t+1])
        cum_totalCommCost[t] = totalCommCost
        if (t+1) % 3000 == 0:
            print('TRIAL:',t,'DONE', '| cum_regret:', cum_regret[t])
            print('Total Communication cost:', totalCommCost)
        # print(cum_regret[t], totalCommCost)
        if cum_regret[t] > break_point:
            print('break at:', t, 'cum. regret:', cum_regret[t])
            break
        
    return dict(A_gob=A_gob, b_gob=b_gob, q_loc=q_loc, c_loc = c_loc, p = p, client_choice = client_choice, r_payoff = r_payoff, totalCommCost=totalCommCost, cum_totalCommCost=cum_totalCommCost)


# In[10]:


oracle_lst = []
true_choice = []
# for case2 set new_y = Y_2
new_y =  -1 * Y_1 + 30
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


# In[11]:


# Initialize q and C

np.random.seed(3) #3 #59
vec_q = np.array([np.random.rand() for _ in range(K * N_FEATURE)])
# vec_q = q[~np.isnan(q)]
# vec_C: C (NK x 1)
np.random.seed(42)
longvec_C = np.array([np.random.rand() for _ in range(N_ARMS * K)])
matrix_c = longvec_C.reshape(N_ARMS, K)
vec_C = matrix_c


# In[12]:


alpha_to_test = [0.75]
print('M:', M)
# for case2 set Y = Y_2.T
results_dict = {alpha: Fed_CLUCB(N_TRIAL=N_TRIAL, N_ARMS=N_ARMS, N_FEATURE=N_FEATURE, break_point = 7500, eta_1 = 0.3, eta_2 = 0.3, alpha_q =alpha, alpha_c = alpha, X=X_1, Y=(-1 * Y_1 + 30 + noise).T, init_q=vec_q, init_c=vec_C,m=M, K = K, X_to_X_m=X_to_X_m, X_reshape=X_reshape, oracle=oracle_case1, gammaU=1.025, iterations=5)                for alpha in alpha_to_test}






