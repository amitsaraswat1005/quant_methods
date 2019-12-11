# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Thu Dec  7 23:41:41 2017
@author: Amit & Sanchi
"""

data_path = 'C:/Users/Amit/RNN_SPX/'


import numpy as np

import pandas as pd

import os
import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
import time as tm

#%% Change of directory
os.chdir(data_path)

##Reading the raw data
data=pd.read_csv('RawData.csv', sep=",")

#%%
##Reading volume data in dataframe
df = pd.DataFrame(data['Volume'])
##Indexing with the date
df.index = data['Ntime']

##Copy of the original dataset
data1 = data.copy()
### Removing  all the price (except open price), volume and time data
data1.drop(['Ntime','time', 'Close Price','Volume','Low Price', 'High Price'], axis=1, inplace=True)

#%%
#Initializing the dataset 
data_df=data1['Open Price'].copy()

#%%
##Calculating DoD returns on the open prices
return_df = (data_df - data_df.shift(1))/data_df
##Convert returns to dataframes
return_df = pd.DataFrame(return_df)

#%%
##Creating a list of lists of returns
returns = return_df.values.tolist()

#%%

##Creating input series for neural networks
column_name = 'Open Price'

# number of data points in the training set
number_batch = 1500

#number of neurons: number of inputs to the h-layer
num_nodes = 2 

## time period after which model parameters were updated to adjust for market dynamics
num_unrollings =  10

# number of days to lookback, length of the input time series
look_back = 20

#a is input returns time series data
a=np.array([return_df[column_name][j:j+look_back].values for j in range(1, len(return_df[column_name])-look_back)],np.float32)

# transaction cost in bps
delta=0.0002

#start_time
time=25

#%%
## number of times neural networks will be trained
num_epochs = 1

## Keeping track of time: Initial time
time1 = tm.time()

## Create a tensorflow graph
g = tf.Graph()

#%%
##Different learning rate functions

## Exponenetial learning rate function
def learning_rate_exponential(rate0, gamma, global_, decay):
    return rate0*gamma**(global_/decay)

## Modified exponential rate function
def modified_learning_rate_exponential(rate0, ratef, number_batch, global_):
    gamma = ratef/rate0
    return rate0 * gamma**((global_+1)/number_batch)

## Constant learning rate function
def constant_learning_rate(rate0):
    return rate0

## Activation functions
## sigmoid function
def sigmoid(x):
    return 1/(1+tf.exp(-x))

## rectified linear unit activation function
def relu_modified(x):
    return (tf.exp(x)-1)

## modified sigmoid function
def sigmoid_modified(w,center,b_):
    return (tf.exp(b_*((w-center)/center))/(1+tf.exp(b_*(w-center)/center))-center)/(1/(1+tf.exp(-b_))-center)

## leaky rectified linear unit function
def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

## hyperbolic tangent function
def tanh(x):
   return 2*sigmoid(2*x)-1 

#%%
def variable_summaries(var):
  ## attach a lot of summaries to a Tensor (for TensorBoard visualization)
  with tf.name_scope('summaries'):
    ## Mean of the series  
    mean = tf.reduce_mean(var)
    ## Summary scalar of the mean
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        ## Standard deviation of the series
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    ## Summary scalar of the standard deviation
    tf.summary.scalar('stddev', stddev)
   
def test_signal(start_date, end_date,column_name,epoch):
    global return_df
    global global_last_signal_value
    signal_convergence = []
    for t in range(start_date, end_date):
        return_component = return_df[column_name][t+1]
        
        if return_component * global_last_signal_value[epoch][t-start_date][0]>0:
            signal_convergence.append(1)
        else :
            signal_convergence.append(0)
    return np.mean(signal_convergence)

with g.as_default():
    # regularization constant
    reg_tf = tf.constant(0.0)
    input_data = []
    for i in range(num_unrollings):
        input_data.append(tf.placeholder(tf.float32,shape=(None,look_back),name='input'))

    #variables
    # input matrix
    U = tf.Variable(tf.truncated_normal([look_back,num_nodes],-0.1,0.1))      
    #recurrent matrix multiplies previous output
    W = tf.Variable(tf.truncated_normal([1,num_nodes],-0.1,0.1))    
    #bias vector
    b = tf.Variable(tf.zeros([1,2*num_nodes]))     
    #output matrix wieghts after the activation function  
    V = tf.Variable(tf.truncated_normal([2*num_nodes,1],-0.1,0.1))
    c = tf.Variable([0.0], [1,1])
    
    #model    
    tensor_real_returns=tf.reshape(input_data[0][0][look_back-1],[1,1]) #appending later, tensor shape preserved
    for i in range(num_unrollings):
        if i == 0:
            output_data_feed=tf.reshape(tf.sign(input_data[0][0][look_back-1]),[1,1])        
            output_ = tf.sign(input_data[0][0][look_back-1])
            a_ = tf.concat((tf.matmul(input_data[i],U),output_*W),axis=1)+b
            h_output = tanh(a_)
            output_after= tanh(tf.matmul(h_output,V)+c)          
        else:
            a_ = tf.concat((tf.matmul(input_data[i],U),output_after*W),axis=1)+b
            h_output = tanh(a_)
            output_after= tanh(tf.matmul(h_output,V)+c)

        output_data_feed=tf.concat((output_data_feed,output_after), axis=0)
        tensor_real_returns=tf.concat((tensor_real_returns,tf.reshape(input_data[i][0][look_back-1],[1,1])),axis=0)

    observed_returns=tf.multiply(output_data_feed[0:-1],tensor_real_returns[1:])-delta*tf.abs(tf.subtract(output_data_feed[1:],output_data_feed[0:-1]))

    #train
    global_step = tf.Variable(0,trainable=False)
    L2_loss = tf.nn.l2_loss(U)
    
    mean_np=tf.reduce_mean(observed_returns)
    mean, variance = tf.nn.moments(observed_returns, axes = [0])
    sharpe = mean/tf.sqrt(variance) 
    R2 = reg_tf*L2_loss
    
    loss = -mean_np

    learning_rate = tf.train.exponential_decay(
        learning_rate=5e3 ,global_step=global_step, decay_steps=5, decay_rate=1, staircase=True)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,var_list=[U,W,V,b,c])

    grad_U=tf.gradients(loss,[U])
    grad_V=tf.gradients(loss,[V])
    grad_W=tf.gradients(loss,[W])
    grad_b=tf.gradients(loss,[b])
    grad_c=tf.gradients(loss,[c])
    
    #optimizer   
    init2 = tf.global_variables_initializer()     
    np.set_printoptions(precision=6)
    sess=tf.Session() 
    sess.run(init2)
    global_ = 0
    global_loss_value = []
    global_sharpe_value = []
    global_last_signal_value = []
    global_last_output_value = []        
    for epoch_i in range((num_epochs)):
        loss_value=[]
        sharpe_value = []
        last_signal_value = []
        last_output_value = []  
        epoch_time = time
    
        for j in range(number_batch): 
            #print('j = ', j)
            rate0 = 5e4 
            ratef = 1e-3
            global_ += 1
            
            for sub_batch in range(1):
                epoch_time=time+j
                feed_input={input_data[i]:a[epoch_time-num_unrollings+i].reshape(1,look_back) for i in range(num_unrollings)}
                temp_sharpe = sess.run(sharpe,feed_dict=feed_input)
                loss_tf_val = sess.run(loss,feed_dict=feed_input)
                last_signal = sess.run(output_data_feed[num_unrollings - 1],feed_dict=feed_input)
                last_output = sess.run(output_after[0][0],feed_dict=feed_input)
                V_prev = sess.run(V,feed_dict=feed_input )
                sess.run(optimizer, feed_dict=feed_input)
                V_after = sess.run(V,feed_dict=feed_input)
                manual_grad_V = (V_after - V_prev)/sess.run(learning_rate, feed_dict = feed_input)
                                
            loss_value.append(loss_tf_val)
            sharpe_value.append(temp_sharpe)
            last_signal_value.append(last_signal)
            last_output_value.append(last_output)
                
        global_loss_value.append(loss_value)
        global_sharpe_value.append(sharpe_value)
        global_last_signal_value.append(last_signal_value)
        global_last_output_value.append(last_output_value)
time2 = tm.time()     

#%%
print(time2 - time1)

#%%
def make_list(input):
    temp = [input[i][0] for i in range(len(input))]
    return temp

#%%
dirty_sharpe_value_dict = {}
i = 0

#%% 
## dirty sharpe value
clean_sharpe_value = [sharpe_value_temp[0] for sharpe_value_temp in global_sharpe_value[i] if sharpe_value_temp <= 2]
dirty_sharpe_value = global_sharpe_value[i]
dirty_sharpe_value = [dirty_sharpe_value[i][0] for i in range(len(dirty_sharpe_value))]
plt.plot(dirty_sharpe_value)

#%% 
## last signal
dirty_last_signal_value = []
for j in range(num_epochs):
    dirty_last_signal_value.append(global_last_signal_value[j])
dirty_last_signal_value = np.array(dirty_last_signal_value).flatten().tolist()
plt.plot(dirty_last_signal_value)

#%% 
## loss
dirty_loss_value = []
for j in range(int(num_epochs)):
    dirty_loss_value.append(global_loss_value[j])
dirty_loss_value = np.array(dirty_loss_value).flatten().tolist()
plt.plot(dirty_loss_value)

#%%
##learning rate
plt.plot([modified_learning_rate_exponential(rate0, ratef, number_batch, i) for i in range(number_batch)])

#%% market_signals
market_signals = [float(np.sign(returns[i])) for i in range(number_batch)]
plt.plot(market_signals)

#%% 
signal_error = [np.abs(market_signals[i] - dirty_last_signal_value[i]) for i in range(1,number_batch)]
plt.plot(signal_error)
print('total signal error = ' , np.sum(signal_error))

#%%
## printing the mean and standard deviation of the series
sharpe_mean = [np.mean(sharpe_series) for sharpe_series in global_sharpe_value]
print(sharpe_mean)
sharpe_std = [np.std(sharpe_series) for sharpe_series in global_sharpe_value]
print(sharpe_std)
#%%
## saving different dataframes
def save_data(path):
    U_df = pd.DataFrame(sess.run(U,feed_dict = feed_input))
    W_df = pd.DataFrame(sess.run(W,feed_dict = feed_input))
    b_df = pd.DataFrame(sess.run(b,feed_dict = feed_input))
    V_df = pd.DataFrame(sess.run(V,feed_dict = feed_input))
    c_df = pd.DataFrame(sess.run(c,feed_dict = feed_input))
    U_df.to_csv(path + 'U.csv')
    W_df.to_csv(path + 'W.csv')
    b_df.to_csv(path + 'b.csv')
    V_df.to_csv(path + 'V.csv')
    c_df.to_csv(path + 'c.csv')    
    return 


#%%
## creating new folder
currentDT = datetime.datetime.now()
print(currentDT.strftime("%m_%d_%H_%M"))
folder_name = currentDT.strftime("%m_%d_%H_%M")

if not os.path.exists(data_path + folder_name):
    os.makedirs(data_path + folder_name)
save_data(data_path + folder_name + '/')

#%%
## save down the data in new path
if (os.path.exists(data_path + folder_name)):
    os.makedirs(data_path + folder_name)
save_data(data_path + folder_name + '/')