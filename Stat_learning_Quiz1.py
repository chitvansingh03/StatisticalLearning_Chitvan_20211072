# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:15:00 2024

@author: ChitvanSingh
"""
# Statistical Learning and Data Science - Quiz -1
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# target function y = x(1-x)
def target_func(x):
    y  = x*(1-x)
    return y 

# hypothesis found by minimising MSE = Ein for h0(x) = b
def  train_hypothesis0(x,X,Y):
    h0_param= np.mean(Y);
    gd_x = h0_param

    return gd_x , h0_param

# hypothesis found by minimising MSE  = Ein for h1(x) = ax + c
def  train_hypothesis1(x,X,Y): 
    (m,) = X.shape
    ones = np.ones((m,))
    X = np.array([X,ones])
    # we will minimize MSE by matrix method: X*h1_param = Y --> h1_param = (inv(X.T*X))*X.T*Y 
    h1_param = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),Y)
    gd2_x = np.dot(np.array([x,1]),h1_param)
    return gd2_x, h1_param


# data points set (a set of [(x1,x2] sampled uniformly from Uniform(0,1)). Note num_lines not to be changed
D_set = np.zeros((2,100)) # num_lines = 100
for i in range(100):
    v = np.random.uniform(0,1,size = 2).reshape(2,)
    D_set[:,i] = v

# Computing gbar : Average of various gd_x we got by minimising different sample points. Averaging of different trainng sets

def avg_D(x, num_lines, train_hypths, targ_func, D_set):
    gd_x = np.zeros((num_lines,1))  # array containing output at 'x' for different hypthesis at diff D's
    gd_func_param = np.zeros((2,num_lines)) # array containing paramters of hypothesis
    for i in range(num_lines):
        v = D_set[:,i]
        w = targ_func(v)
        gdx , param = train_hypths(x, v ,w )
        gd_x[i] = gdx
        gd_func_param[:,i] = param
    
    gbar_param = np.mean(gd_func_param,axis = 1)
    print(gbar_param.shape,gd_x.shape)
    # putting this condition so i can compute bias and variance for both the hypthesis classes
    if len(gbar_param)== 2:
        gbar_x = np.dot(np.array([x,1]).reshape(2,) , gbar_param)
        
    else: 
        gbar_x = gbar_param[0]
    
    print(gbar_x.shape,gd_x.shape)
    variance_x = (np.matmul((gd_x - gbar_x).T , (gd_x - gbar_x)))/num_lines  # Variance(x)
    bias_x = (gbar_x - target_func(x))**2  # Bias(x)
    return gbar_param,variance_x , bias_x , gd_func_param


num_x_samples = 1000  # no. of x from [0,1)
x_data = np.random.uniform(0,1 , size  = num_x_samples)  # generating unofrom data from [0,1]


# Computing Variance and bias over data points, as well as out of sample error

def Calc_Eout_bias_var(x_data,num_x_samples, num_lines , train_hypts , targ_func , D_set):
    # following arrays are diffined to contain the values for all x's in x_data. Eg: biases contains bias for all x in x_data
    biases = np.zeros((num_x_samples,1))
    variances = np.zeros((num_x_samples,1))
    lowerbd = np.zeros((num_x_samples,1))  # lower bound of 1 std.dev. lb(x) = gbar(x)  - sqrt(variance(x))
    upperbd = np.zeros((num_x_samples,1))  # upper bound of 1 std.dev. lb(x) = gbar(x)  + sqrt(variance(x))
    for k in range(len(x_data)):
        gbar_parm ,variance ,bias, gd_func_param  = avg_D(x_data[k], num_lines, train_hypts , targ_func, D_set)
        variances[k] = variance
        biases[k] = bias
        if len(gbar_parm)== 2:
            gbar_x = np.dot(np.array([x_data[k],1]).reshape(2,) , gbar_parm)
            
        else: 
            gbar_x = gbar_parm
        lowerbd[k] = gbar_x - np.sqrt(variance)
        upperbd[k] = gbar_x + np.sqrt(variance)
        
    var = (sum(variances))/num_x_samples # variance 
    bias = (sum(biases))/num_x_samples  # bias
    
    # Out of sample error is  = bias + Variance
    Eout = var + bias  
    gd_param = gd_func_param
    
    return Eout, var , bias, gd_param ,lowerbd, upperbd , gbar_parm


# Calculating values for H1 and H0 classes for num_lines = 100, num_samples  1000
Eout1 , variance1 , bias1, gd_param1 , lowerbd1 , upperbd1, gbarparm1 = Calc_Eout_bias_var(x_data, 1000, 100, train_hypothesis1, target_func, D_set)
Eout0 , variance0 , bias0, gd_param0 , lowerbd0 , upperbd0, gbarparm0 = Calc_Eout_bias_var(x_data, 1000, 100, train_hypothesis0, target_func, D_set)


ones = np.ones((1000,))
xfor_gbar = np.array([x_data,ones]).T

#plotting

# Plotting for H1: target fnction, g_bar, upper and llower bounds
plt.plot(x_data , target_func(x_data) , 'r.' , label = 'f(x) = x(1-x)')
plt.plot(x_data , np.matmul(xfor_gbar,gbarparm1) , 'b' , label = 'Final hypothesis y = -.031x + 0.19')
plt.plot(x_data , lowerbd1 , 'g' , label = 'lower bound of 1 std.dev')
plt.plot(x_data , upperbd1 , 'c' , label = 'lower bound of 1 std.dev')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'upper right', bbox_to_anchor = (1,1))
plt.show()


# Plotting for H0: target fnction, g_bar, upper and llower bounds
plt.plot(x_data , target_func(x_data) , 'r.' , label = 'f(x) = x(1-x)')
plt.plot(x_data , 0.1651804*np.ones((1000,)) , 'b' , label = 'Final hypothesis y =0.1651')
plt.plot(x_data , lowerbd0 , 'g' , label = 'lower bound of 1 std.dev')
plt.plot(x_data , upperbd0 , 'c' , label = 'lower bound of 1 std.dev')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'upper right', bbox_to_anchor = (1,1))
plt.show()

# Just the target funtion
plt.plot(x_data , target_func(x_data) , 'r.' , label = 'f(x) = x(1-x)')

#Q1: Finding MSE best fits 

#H1: Using the matrix method to minimize least sqaures
G = np.matmul(xfor_gbar.T,xfor_gbar )
Ginv = np.linalg.inv(G)
Y = target_func(x_data).reshape(1000,1)
K = np.matmul(Ginv,xfor_gbar.T)
model_param = np.matmul(K,Y)

Y_pred = np.matmul(xfor_gbar,model_param)
MSE = np.matmul((Y_pred - Y).T , (Y_pred - Y))/1000 # Residual

# Plotting for best fit MSE H1
plt.plot(x_data , target_func(x_data) , 'r.' , label = 'f(x) = x(1-x)')
plt.plot(x_data , Y_pred , 'g.' , label = 'f(x) = 0.014x + 0.15 , MSE = 0.005',)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'upper right', bbox_to_anchor = (1,1))
plt.show()

#H0 : it is just the mean of all values
b = np.mean(Y)
MSE0 = np.matmul((Y - b).T,(Y-b))/1000
plt.plot(x_data , target_func(x_data) , 'r.' , label = 'f(x) = x(1-x)')
plt.plot(x_data , b*np.ones((1000,1)) , 'g.' , label = 'f(x) = 0.0163 , MSE = 0.005')
plt.title('Hypothesis class 0 best fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'upper right', bbox_to_anchor = (1,1))
plt.show()

#Q2 Plotting
# plotting hypothesis MSE in sample minimized outputs based on D (traning set)
#H0: y = b
for j in range(100):
    plt.plot(x_data ,(gd_param0[0,j])*np.ones((1000,)), color = (j/500,j/400,0))
plt.plot(x_data , 0.1651804*np.ones((1000,)) , 'b' , label = 'Final hypothesis y =0.1651')  # plotting average hypothesis for comparision 
plt.plot(x_data , target_func(x_data) , 'r.' , label = 'f(x) = x(1-x)')  # plotting target function for comparision
plt.title('Hypotheis class 0 w.r.t D')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'upper right', bbox_to_anchor = (1,1))
plt.show()
 

#H1: y = mx + c   
for j in range(100):
    plt.plot(x_data , np.matmul(xfor_gbar , gd_param1[:,j]) , color = (j/500,j/400,j/300))

plt.plot(x_data , np.matmul(xfor_gbar,gbarparm1) , 'b' , label = 'Final hypothesis y = -.031x + 0.19') # plotting average hypothesis for comparision
plt.plot(x_data , target_func(x_data) , 'r.' , label = 'f(x) = x(1-x)')    # plotting target function for comparision
plt.title('Hypotheis class 1 w.r.t D')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'upper right', bbox_to_anchor = (1,1))
plt.show()
             





        
        
        




    

