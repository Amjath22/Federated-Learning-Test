import numpy as np
import pandas as pd
import tensorflow as tf
import flwr as fl
import sys

data=pd.read_csv('Churn_Modelling1_5000.csv')
# no missing data########

#spiting the dataframe into two

X=data.iloc[:,3:-1].values
Y=data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder

labEnc=LabelEncoder() #gender column
X[:,2]=labEnc.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer #geography column
from sklearn.preprocessing import OneHotEncoder
ColTrans=ColumnTransformer(transformers=[('encoder',OneHotEncoder(sparse=False),[1])],remainder='passthrough')
X=np.array(ColTrans.fit_transform(X));

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation='relu')) #hidden layer 1
ann.add(tf.keras.layers.Dense(units=6,activation='relu')) #hidden layer 2
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid')) #output layer

ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self,parameters,config):
        model.set_weights(parameters)
        #ann.fit(X_train,Y_train, batch_size=32, epochs=25) #Training the model
        return ann.get_weights()

    def evaluate(self,parameters,config):
        model.set_weights(parameters)
        loss,accuracy=model.evaluate(X_test,Y_test, verbose=0)
        print("Evaluation accuracy: ",accuracy)
        return loss,{"accuracy":accuracy}

fl.client.start_numpy_client(server_address='localhost:'+str(sys.argv[1]),client=FlowerClient(),grpc_max_message_length=1024*1024*1024)
















