#numpy kullandım çünkü exp ve log gibi işemlerde numpy kullanmaduğım koşullarda hata aldım ve çözüm bulamadım

import numpy as np

class LogisticRegression:

    def __init__(self,learning_rate,epoch,batch_size):
        global batch_s
        batch_s = batch_size
        global learning_r
        learning_r = learning_rate
        global ep
        ep = epoch
        pass
    def fit(self,x_train, y_train, x_test, y_test,data):
        dimension =  x_train.shape[0]
        w,b = self.initialize_weights_and_bias(dimension)
        parameters, gradients, cost_list = self.update(w, b, x_train, y_train, x_test,y_test,data,learning_r,ep)
        return parameters
    
    def predict(self,y_test,x_test,parameters):
        y_prediction_test = self.predict_f(parameters["weight"],parameters["bias"],x_test)
        
        print(y_prediction_test)
    
        print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    def create_mini_batches(self,X, y,data, batch_size): 
        mini_batches = []
        #data = X
        #np.random.shuffle(data) 
        n_minibatches = data.shape[0] // batch_size 
        i = 0
      
        for i in range(n_minibatches + 1): 
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
            mini_batches.append((X_mini, Y_mini)) 
        if X.shape[0] % batch_size != 0: 
            mini_batch = data[i * batch_size:data.shape[0]] 
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
            mini_batches.append((X_mini, Y_mini)) 
        return mini_batches 
      
    
    
    def initialize_weights_and_bias(self,dimension):
        
        w = np.full((dimension,1),0.01)
        b = 0.0
        return w,b
    
    
    def softmax(self,x):
    
        max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
        e_x = np.exp(x - max) #subtracts each row with its max value
        sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
        f_x = e_x / sum 
        return f_x

    
    def forward_backward_propagation(self,w,b,x_train,y_train):
    
        z = np.dot(w.T,x_train) + b
        y_head = self.softmax(z)
        loss = -y_train*np.log(y_head,where= y_head > 0)
        cost = (np.sum(loss))/x_train.shape[1] 
        
    
        derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] 
        derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 
        gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
        
        return cost,gradients
    
    
    def update(self,w, b, x_train, y_train, x_test, y_test,data,learning_rate,number_of_iterarion):
        cost_list = []
        cost_list2 = []
        accuracy_list = []
        index = []
        
        for i in range(number_of_iterarion):
            mini_batches = self.create_mini_batches(x_train, y_train,data, batch_size = batch_s) 
            for mini_batch in mini_batches:
                cost,gradients = self.forward_backward_propagation(w,b,x_train,y_train)
                cost_list.append(cost)
                X_mini, y_mini = mini_batch 
                w = w - learning_rate * gradients["derivative_weight"]
                b = b - learning_rate * gradients["derivative_bias"]
    
            if i % 10 == 0:
                cost_list2.append(cost)
                index.append(i)    
                y_prediction_test = self.predict_f(w,b,x_test)
                accuracy = (100 - np.mean(np.abs(y_prediction_test - y_test)) * 100)
                accuracy_list.append(accuracy)
                
        parameters = {"weight": w,"bias": b}
        return parameters, gradients, cost_list
    
    def predict_f(self,w,b,x_test):
        z = self.softmax(np.dot(w.T,x_test)+b)
        Y_prediction = np.zeros((1,x_test.shape[1]))
        for i in range(z.shape[1]):
            if z[0,i]<= 0.5:
                Y_prediction[0,i] = 0
            else:
                Y_prediction[0,i] = 1
    
        return Y_prediction


        
    #logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 300) 