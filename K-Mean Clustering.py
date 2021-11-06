#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.show()


# In[50]:


pwd


# In[51]:


df = pd.read_csv("C:/Users/RUSHIKESH SUNIL/Documents/Internship/customers.csv")
df.head()


# In[31]:


df.shape


# In[32]:


df.isnull().sum()


# In[37]:


X = df.iloc[:,[3,4]].values
X


# In[40]:


plt.scatter(X[...,0],X[...,1])
plt.xlabel('Total Income')
plt.ylabel('Spending Score')
plt.show


# In[43]:


from sklearn.cluster import KMeans
wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    


# In[44]:


wcss


# In[56]:


plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Cluster")
plt.ylabel("WCSS")
plt.show()


# In[58]:


KMeans(n_clusters=5,init='k-means++',random_state=0)
Y_kmeans = kmeans.fit_predict(X)
Y_kmeans


# In[62]:


X[Y_kmeans==0,0]


# In[63]:


X[Y_kmeans==0,1]


# In[79]:


plt.scatter(X[Y_kmeans==0,0],X[Y_kmeans==0,1],label = 'Careless')
plt.scatter(X[Y_kmeans==1,0],X[Y_kmeans==1,1],label = 'Middle Class')
plt.scatter(X[Y_kmeans==2,0],X[Y_kmeans==2,1],label = 'Target')
plt.scatter(X[Y_kmeans==3,0],X[Y_kmeans==3,1],label = 'Smart')
plt.scatter(X[Y_kmeans==4,0],X[Y_kmeans==4,1],label = 'low ')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


# In[85]:


df['Target'] = Y_kmeans


# In[83]:


df


# In[ ]:




