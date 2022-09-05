#!/usr/bin/env python
# coding: utf-8

# In[36]:


#------------------------------------------------------
#            Start               
#------------------------------------------------------


# In[37]:


#------------------------------------------------------
#             Import Libary                  
#------------------------------------------------------
import matplotlib.pyplot as plt
import numpy  as np
import math
import pandas as pd
import time   as ti
import statsmodels.api as sm
import seaborn as sns
sns.set()
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import networkx as nx
from sklearn.neighbors import NearestNeighbors
#------------------------------------------------------


# In[38]:


#------------------------------------------------------
# Define For Select Row Of Matrix                    
#------------------------------------------------------
def Index_Select(matrix, i):
    return [row[i] for row in matrix]
#------------------------------------------------------


# In[39]:


def KNN(Matrix_Graph,K,Kind):
    knn = NearestNeighbors(algorithm='auto',  n_neighbors=K, p=2);
    knn.fit(Matrix_Graph);
    distances, indices = knn.kneighbors(Matrix_Graph);
    if Kind==0:
        return distances
    if Kind==1:
        return  indices


# In[40]:


def W(X,K,sigma):
    n=len(X);
    W = np.zeros((n, n));
    distances =KNN(X,K,0)
    indices   =KNN(X,K,1)
    for i in range(n):
        b=indices[i];
        for j in range(K):
            W[i,b[j]] = math.exp(-((distances[i,j])**2)/(sigma**2));
            W[b[j],i] = math.exp(-((distances[i,j])**2)/(sigma**2));
    return W


# In[41]:


def Chech_Zero(Number):
    return max(Number,10**(-8))


# In[42]:


def Diagonal_Matrix(First_Matrix):
    D_Norm_List=[]
    for i in range(0,len(First_Matrix)):
                   D_Norm= np.linalg.norm(First_Matrix[i])
                   D_Norm_List.append(D_Norm)
    return np.diag(D_Norm_List)


# In[43]:


def U_Value(X,V,S):
    E=X-(np.dot(X,np.dot(S,V)))
    E_Norm_List=[]
    for i in range(0,len(E)):
        U_Norm          = np.linalg.norm(E[i])
        U_ii=1/ Chech_Zero(U_Norm)
        E_Norm_List.append(U_ii)
    return np.diag(E_Norm_List)


# In[44]:


#------------------------------------------------------
#             Definision MFFS                  
#------------------------------------------------------
def SLSDR(X,K,Steps,Alpha,Beta,Lambda,Break_Value,sigma):

    #-----------
    #Inter Parametr And Need Value
    #-----------
    X                   =  np.array(X)
    Y                   =  np.array(X)
    Row_Number          =  len(X)
    Column_Number       =  len(X[0])
    S                   =  np.random.rand(Column_Number,K)
    V                   =  np.random.rand(K,Column_Number)
    S_Final_Norm        =  []
    S_Final_Norm_Index  =  []
    #-----------
    #End
    #-----------
    
    
    #-----------  
    #Start Algoritm
    #-----------
    Start=ti.time()
    for i in range(Steps):
        
        
        #-----------
       
        #-----------
        
        
        #-----------
        W_S=W(X,K,sigma)
        #-----------
        D_S=Diagonal_Matrix(W_S)
        
        
        #-----------
        W_V=W(X.T,K,sigma)
        #-----------
        D_V=Diagonal_Matrix(W_V)
    
    
        #----------- 
        U=U_Value(X,V,S)
        #----------- 
        
        
        #----------- 
        S_UP=np.dot(np.dot(np.dot(X.T,U),X),V.T) +   np.dot(Alpha * (np.dot(np.dot(X.T,W_S),X)) + ((np.identity(Column_Number))
       *  (Beta + Lambda)   ),S) 
        
        
        S_DOWN1=np.dot(np.dot(np.dot(np.dot(np.dot(X.T,U),X),S),V),V.T) 
        S_DOWN2=np.dot((Alpha*(np.dot(np.dot(X.T,D_S),X))) + (Beta*np.ones(Column_Number))+ Lambda *np.dot(S,S.T),S)
        S_DOWN=S_DOWN1+S_DOWN2
        
        S=S*(S_UP/S_DOWN)
        #----------- 
 
        
        
        
        
        
        
        
        #----------- 
        V_UP=np.dot(np.dot(np.dot(S.T,X.T),U),X) +   Alpha*(np.dot(V,W_V))
        
        
        V_DOWN=np.dot(np.dot(np.dot(np.dot(np.dot(S.T,X.T),U),X),S),V)  +  Alpha*(np.dot(V,D_V))
        
                                                                                  
        V=V*(V_UP/V_DOWN)   
        #----------- 
        
        
        
        
        
        
        #-----------
    #Calculate Norm OF W And Sort Index And Norm
    #-----------
    for i in range(0,Column_Number):
        S_Norm          = np.linalg.norm(S[i])
        S_Final_Norm.append(S_Norm)
        S_Final_Norm_Index.append(i+1)
        
        
    S_Final_Norm        = np.array(S_Final_Norm)
    S_Final_Norm_Index  = np.array(S_Final_Norm_Index)
    S_Norm_Index        = np.array([[S_Final_Norm],[S_Final_Norm_Index]])
    S_Norm_Index        = S_Norm_Index.T
    S_Norm_Index        = np.matrix(S_Norm_Index)
    S_Sorted            = S_Norm_Index[np.argsort(S_Norm_Index.A[:, 0])]
    S_Sorted            = np.array(S_Sorted)
    Final_Index         = Index_Select(S_Sorted,1)
    Final_Index.reverse()
    S_Sorted            = np.array(S_Sorted)
    Final_Norm          = Index_Select(S_Sorted,0)
    Final_Norm.reverse()
    #-----------  
    #End
    #----------- 
    
    
    
    #-----------  
    #Select Need Dimension Of Main Dataset  And Show    
    #-----------
    
    Dimension_Index_List=[]
    Final_Index_List=[]
    for i in range(K):
        My_Index=int(Final_Index[i]-1)
        Dimension_Index_List.append(My_Index)
        Final_Index_List.append(int(Final_Index[i]))
    Data_Set_Main=np.array(Y.T)
    Selected_Column=Data_Set_Main[Dimension_Index_List]
    return Selected_Column.T
    #-----------  
    #End  
    #-----------
#------------------------------------------------------
        
        
        
        
        
        
        


# In[45]:


Main_Data_Set = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
    ]

SLSDR(np.mat(Main_Data_Set).T,1,10,3,0.1,0.1,0,1)


# In[46]:


#------------------------------------------------------
#             Definishion K_Means Algoritm (Clustering)                 
#------------------------------------------------------
def K_Means_Clustering(Data_Set,Count_Of_Cluster):
    
    kmeans = KMeans(Count_Of_Cluster)
    kmeans.fit(Data_Set)
    identified_clusters = kmeans.fit_predict(Data_Set)
    kmeans=pd.DataFrame(identified_clusters, index= None)
    kmeans_Label=np.matrix(kmeans)
    return kmeans_Label
#------------------------------------------------------


# In[47]:


#------------------------------------------------------
#             Definishion accuracy                 
#------------------------------------------------------
def Acc(Main_Labels,K_Labels):
    
    P=Main_Labels
    Q=np.array(K_Labels).ravel().tolist()
    
    return accuracy_score(P,Q)
#------------------------------------------------------


# In[48]:


#------------------------------------------------------
#             Definishion normalized mutualinformation              
#------------------------------------------------------
def NMI(Main_Labels,K_Labels):
    
    P=Main_Labels
    Q=np.array(K_Labels).ravel().tolist()
    I_PQ=mutual_info_score(P,Q)
    H_P=entropy(P)
    H_Q=entropy(Q)
    
    return I_PQ/((H_P*H_Q)**(1/2))
#------------------------------------------------------


# In[49]:


#------------------------------------------------------
#             Calculate Count Of Cluster          
#------------------------------------------------------
def Cluster_Count(Main_Labels):
    
    input_list=Main_Labels
    l1 = []
    count = 0
    for item in input_list:
        if item not in l1:
            count += 1
            l1.append(item)
    Cluster_Count=count
    return Cluster_Count


# In[50]:


def Multi_SLSDR(K):
    #Data_Set=SLSDR(np.mat(Main_Data_Set).T,K,Steps,Alpha,Beta,Lambda,Break_Value,sigma)
    Data_Set=SLSDR((Main_Data_Set),K,Steps,Alpha,Beta,Lambda,Break_Value,sigma)
    return  Data_Set


# In[51]:


def result(Main_Labels,Dimension_Select_List,Kmeans_Count,sigma):
    Multi_ACC_List  =[]
    Multi_NMI_List  =[]
    ACC_List        =[]
    NMI_List        =[]
    ACC_Std         =[]
    NMI_Std         =[]
    for i in Dimension_Select_List:
        Data_Set=Data_Sets(i)
        for j in range(Kmeans_Count):
            K_Means =K_Means_Clustering(Data_Set,Cluster_Count(Main_Labels))
            K_Labels=K_Means
            Multi_ACC_List.append(Acc(Main_Labels,K_Labels))
            Multi_NMI_List.append(NMI(Main_Labels,K_Labels))
        ACC_List.append(np.mean(Multi_ACC_List))
        ACC_Std.append(np.std(Multi_ACC_List))
        NMI_List.append(np.mean(Multi_NMI_List))
        NMI_Std.append(np.std(Multi_NMI_List))
        Multi_ACC_List.clear()
        Multi_NMI_List.clear()
    #print(ACC_Std)
    #print(NMI_Std)

    
    
    #----------- 
    #Show ACC in 2*D
    #-----------
    X_List=Dimension_Select_List
    Y_List=ACC_List
    plt.plot(X_List,Y_List,color='lightcoral',marker='D',markeredgecolor='black')
    plt.ylim(0,1) 
    plt.xlim(0,10) 
    plt.xlabel('Count Of Dimension') 
    plt.ylabel('Acc Value') 
    plt.title('ACC') 
    plt.show()
    #-----------  
    #End
    #-----------  
    
    
    
    #----------- 
    #Show Show NMI in 2*D
    #-----------
    X_List=Dimension_Select_List
    Y_List=NMI_List
    plt.plot(X_List,Y_List,color='lightcoral',marker='D',markeredgecolor='black')
    plt.ylim(0,1) 
    plt.xlim(0,10) 
    plt.xlabel('Count Of Dimension') 
    plt.ylabel('NMI Value') 
    plt.title('NMI') 
    plt.show()
    #-----------  
    #End
    #-----------


# In[52]:


def Data_Sets(K):
    return Multi_SLSDR(K)
    #return Main_Data_Set


# In[55]:


Main_Data_Set = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [1,0,0,4],
    ]
Steps=10
Alpha=0.01
Beta=.02
Lambda=2
Break_Value=4


Main_Labels =[1,0,0,2,0]


Dimension_Select_List=[1,2,3,4]

Kmeans_Count=5

sigma=1

result(Main_Labels,Dimension_Select_List,Kmeans_Count,sigma)


# In[56]:


Main_Data_Set =pd.read_csv('C:\\Users\\babak_Nouri\\Desktop\\Cluster_Data.CSV', header=None, skiprows=1)

Labels = pd.read_csv('C:\\Users\\babak_Nouri\\Desktop\\Cluster_Label.CSV', header=None, skiprows=1)
Main_Labels=np.array(Labels).ravel().tolist()


Steps=10
Alpha=0.01
Beta=.02
Lambda=2
Break_Value=4



Dimension_Select_List=[1,2,3,4]

Kmeans_Count=5

sigma=1

result(Main_Labels,Dimension_Select_List,Kmeans_Count,sigma)


# In[ ]:





# In[ ]:




