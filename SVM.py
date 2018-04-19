
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import SVR


# In[ ]:


filepath='op.json'
df=pd.read_json(filepath,lines=True)
df['ioreadTime']=df['duration'] - df['executorCpuTime']
df=df[['schedulerDelay','executorCpuTime','resultSize','bytesRead','duration','ioreadTime','executorRunTime']]


# In[ ]:


X = df.iloc[:,1:7].values
Y = df.iloc[:,len(df.iloc[0])-1].values
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
print(X_test)


# In[ ]:


clf = SVR(kernel='rbf')
clf.fit(X_train,y_train) 


# In[ ]:


y_pred = clf.predict(X_test)
print(y_pred)
print(accuracy_score(y_test, y_pred,normalize=True))

