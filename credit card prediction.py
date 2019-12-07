#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


os.getcwd


# In[3]:


os.chdir('F:\\data science\\DataScience-With-Python-master\\DataScience-With-Python-master\\Classification\\')


# In[4]:


dataset=pd.read_csv('Python_Credit_Risk_XTrain.csv')


# In[5]:


dataset


# In[6]:


dataset.isnull().sum()


# In[7]:


dataset.isnull().sum()/len(dataset)*100


# In[8]:


dataset.info()


# In[9]:


sns.boxplot(y='LoanAmount',data=dataset)


# In[10]:


dataset.groupby('Gender').size()


# In[11]:


dataset['Gender']=dataset['Gender'].fillna('Male')


# In[12]:


dataset['Gender']=dataset['Gender'].astype('category')
dataset['Gender']=dataset['Gender'].cat.codes


# In[13]:


dataset.groupby('Married').size()


# In[14]:


dataset['Married']=dataset['Married'].fillna('Yes')


# In[16]:


dataset['Married']=dataset['Married'].astype('category')
dataset['Married']=dataset['Married'].cat.codes


# In[17]:


dataset.groupby('Dependents').size()


# In[18]:


dataset['Dependents']=dataset['Dependents'].fillna('0')


# In[19]:


dataset['Dependents']=dataset['Dependents'].astype('category')
dataset['Dependents']=dataset['Dependents'].cat.codes


# In[20]:


dataset.groupby('Self_Employed').size()


# In[21]:


dataset['Self_Employed']=dataset['Self_Employed'].fillna('No')


# In[22]:


dataset['Self_Employed']=dataset['Self_Employed'].astype('category')
dataset['Self_Employed']=dataset['Self_Employed'].cat.codes


# In[23]:


dataset.groupby('Loan_Amount_Term').size()


# In[24]:


dataset['Loan_Amount_Term']=dataset['Loan_Amount_Term'].fillna(360)


# In[25]:


dataset['Loan_Amount_Term']=dataset['Loan_Amount_Term'].astype('category')
dataset['Loan_Amount_Term']=dataset['Loan_Amount_Term'].cat.codes


# In[26]:


dataset.groupby('Credit_History').size()


# In[27]:


dataset['Credit_History']=dataset['Credit_History'].fillna(1.0)


# In[28]:


dataset['Education']=dataset['Education'].astype('category')
dataset['Education']=dataset['Education'].cat.codes


# In[30]:


dataset['Property_Area']=dataset['Property_Area'].astype('category')
dataset['Property_Area']=dataset['Property_Area'].cat.codes


# In[31]:


dataset['Loan_Status']=dataset['Loan_Status'].astype('category')
dataset['Loan_Status']=dataset['Loan_Status'].cat.codes


# In[32]:


dataset['LoanAmount']=dataset['LoanAmount'].fillna(dataset['LoanAmount'].median())


# In[33]:


dataset.isnull().sum()


# In[34]:


dataset.isnull().sum()/len(dataset)*100


# In[35]:


sns.boxplot(y='LoanAmount',data=dataset)


# In[36]:


dataset


# In[37]:


x=dataset.iloc[:,1:12].values


# In[38]:


y=dataset.iloc[:,-1].values


# In[40]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x=sc_x.fit_transform(x)


# # logistics regression

# In[42]:


from sklearn.linear_model import LogisticRegression
logmodel =LogisticRegression()
logmodel.fit(x,y)


# # load test data

# In[76]:


test = pd.read_csv('Python_Module_Day_15.4_Credit_Risk_Validate_data_XTEST.csv')


# In[77]:


test


# In[78]:


test.isnull().sum()/len(test)*100


# In[79]:


test.groupby('Gender').size()


# In[80]:


test['Gender']=test['Gender'].fillna('Male')


# In[81]:


test['Gender']=test['Gender'].astype('category')
test['Gender']=test['Gender'].cat.codes


# In[82]:


test.groupby('Dependents').size()


# In[83]:


test['Dependents']=test['Dependents'].fillna('0')


# In[84]:


test['Dependents']=test['Dependents'].astype('category')
test['Dependents']=test['Dependents'].cat.codes


# In[85]:


test.groupby('Self_Employed').size()


# In[86]:


test['Self_Employed']=test['Self_Employed'].fillna('No')


# In[87]:


test['Self_Employed']=test['Self_Employed'].astype('category')
test['Self_Employed']=test['Self_Employed'].cat.codes


# In[88]:


test.groupby('Loan_Amount_Term').size()


# In[89]:


test['Loan_Amount_Term']=test['Loan_Amount_Term'].astype('category')
test['Loan_Amount_Term']=test['Loan_Amount_Term'].cat.codes


# In[90]:


test.groupby('Loan_Amount_Term').size()


# In[91]:


test['Loan_Amount_Term']=test['Loan_Amount_Term'].fillna(10)


# In[92]:


test.groupby('Loan_Amount_Term').size()


# In[93]:


test.groupby('Credit_History').size()


# In[94]:


test['Credit_History']=test['Credit_History'].fillna(1.0)


# In[95]:


sns.boxplot(y='LoanAmount',data=test)


# In[96]:


test['LoanAmount']=test['LoanAmount'].fillna(test['LoanAmount'].median())


# In[97]:


test.isnull().sum()/len(test)*100


# In[98]:


test


# In[99]:


test['Education']=test['Education'].astype('category')
test['Education']=test['Education'].cat.codes


# In[100]:


test['Married']=test['Married'].astype('category')
test['Married']=test['Married'].cat.codes


# In[101]:


test['Property_Area']=test['Property_Area'].astype('category')
test['Property_Area']=test['Property_Area'].cat.codes


# In[103]:


test['outcome']=test['outcome'].astype('category')
test['outcome']=test['outcome'].cat.codes


# In[104]:


test


# In[105]:


x_test=test.iloc[:,1:12]


# In[106]:


x_test


# In[107]:


y_test=test.iloc[:,-1]


# In[108]:


y_test


# In[109]:


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_test=sc_x.fit_transform(x_test)


# In[110]:


y_pred=logmodel.predict(x_test)


# In[111]:


y_pred


# In[115]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


# In[116]:


cm


# In[117]:


(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # K nearest neighbours

# In[118]:


from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=5,metric='euclidean',p=2)
classifier_knn.fit(x,y)


# In[119]:


y_pred=classifier_knn.predict(x_test)


# In[120]:


cm=confusion_matrix(y_test,y_pred)


# In[121]:


cm


# In[122]:


(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # naive bayes

# In[125]:


from sklearn.naive_bayes import GaussianNB
classifier_nb=GaussianNB()
classifier_nb.fit(x,y)


# In[126]:


y_pred=classifier_nb.predict(x_test)


# In[127]:


cm=confusion_matrix(y_test,y_pred)


# In[128]:


(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # support vetor machine:Linear Kernel

# In[129]:


from sklearn.svm import SVC
classifier_svm_linear=SVC(kernel='linear')
classifier_svm_linear.fit(x,y)


# In[130]:


y_pred=classifier_svm_linear.predict(x_test)


# In[131]:


cm=confusion_matrix(y_test,y_pred)


# In[132]:


(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # support vector machine:sigmoid kernel

# In[133]:


from sklearn.svm import SVC
classifier_svm_sigmoid = SVC(kernel='sigmoid')
classifier_svm_sigmoid.fit(x,y)


# In[134]:


y_pred=classifier_svm_sigmoid.predict(x_test)


# In[135]:


cm=confusion_matrix(y_test,y_pred)


# In[136]:


cm


# In[137]:


(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # support vector machine:radial basis function kernel

# In[138]:


from sklearn.svm import SVC
classifier_svm_ref = SVC(kernel='rbf')
classifier_svm_ref.fit(x,y)


# In[139]:


y_pred=classifier_svm_ref.predict(x_test)


# In[140]:


cm=confusion_matrix(y_test,y_pred)


# In[141]:


cm


# In[142]:


(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # support vector machine:polynomial function kernel

# In[143]:


from sklearn.svm import SVC
classifier_svm_poly=SVC(kernel='poly')
classifier_svm_poly.fit(x,y)


# In[145]:


y_pred=classifier_svm_poly.predict(x_test)


# In[146]:


cm=confusion_matrix(y_test,y_pred)


# In[147]:


cm


# In[148]:


(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # decision tree

# In[149]:


from sklearn.tree import DecisionTreeClassifier
classifier_dt=DecisionTreeClassifier(criterion='entropy')
classifier_dt.fit(x,y)


# In[150]:


y_pred=classifier_dt.predict(x_test)


# In[151]:


cm=confusion_matrix(y_test,y_pred)


# In[152]:


cm


# In[153]:


(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # Random forest

# In[155]:


from sklearn.ensemble import RandomForestClassifier
classifier_rf=RandomForestClassifier(n_estimators=3,criterion='entropy')
classifier_rf.fit(x,y)


# In[156]:


y_pred=classifier_rf.predict(x_test)


# In[157]:


cm=confusion_matrix(y_test,y_pred)


# In[158]:


cm


# In[159]:


(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,0]+cm[0,1]+cm[1,1])*100


# # validate data

# In[172]:


validation=pd.read_csv('Python_Module_Day_15.3_Credit_Risk_Test_data.csv')


# In[173]:


validation


# In[174]:


validation.isnull().sum()


# In[175]:


validation.isnull().sum()/len(validation)*100


# In[176]:


validation.groupby('Gender').size()


# In[177]:


validation.groupby('Dependents').size()


# In[178]:


validation.groupby('Self_Employed').size()


# In[179]:


validation.groupby('Loan_Amount_Term').size()


# In[180]:


validation.groupby('Credit_History').size()


# In[181]:


sns.boxplot(y='LoanAmount',data=validation)


# In[170]:





# In[183]:


validation.groupby('Gender').size()


# In[182]:


validation['Gender']=validation['Gender'].fillna('Male')


# In[184]:


validation['Gender']=validation['Gender'].astype('category')
validation['Gender']=validation['Gender'].cat.codes


# In[185]:


validation['Dependents']=validation['Dependents'].fillna('0')
validation['Dependents']=validation['Dependents'].astype('category')
validation['Dependents']=validation['Dependents'].cat.codes


# In[186]:


validation['Self_Employed']=validation['Self_Employed'].fillna('No')
validation['Self_Employed']=validation['Self_Employed'].astype('category')
validation['Self_Employed']=validation['Self_Employed'].cat.codes


# In[188]:


validation['LoanAmount']=validation['LoanAmount'].fillna(validation['LoanAmount'].median())


# In[189]:


validation['Loan_Amount_Term']=validation['Loan_Amount_Term'].fillna(360)


# In[190]:


validation['Credit_History']=validation['Credit_History'].fillna(1.0)


# In[192]:


validation.isnull().sum()


# In[193]:


validation['Married']=validation['Married'].astype('category')
validation['Married']=validation['Married'].cat.codes


# In[194]:


validation['Education']=validation['Education'].astype('category')
validation['Education']=validation['Education'].cat.codes


# In[195]:


validation['Property_Area']=validation['Property_Area'].astype('category')
validation['Property_Area']=validation['Property_Area'].cat.codes


# In[196]:


validation


# In[197]:


x_val=validation.iloc[:,1:12].values


# In[198]:


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_val=sc_x.fit_transform(x_val)


# # applying model in the validate data

# In[199]:


y_pred=logmodel.predict(x_val)


# In[200]:


y_pred


# In[201]:


y_pred=classifier_knn.predict(x_val)


# In[202]:


y_pred


# In[203]:


y_pred=classifier_nb.predict(x_val)


# In[204]:


y_pred


# In[205]:


y_pred=classifier_svm_linear.predict(x_val)


# In[206]:


y_pred


# In[208]:


y_pred=classifier_svm_poly.predict(x_val)


# In[209]:


y_pred


# In[210]:


y_pred=classifier_svm_ref.predict(x_val)
y_pred


# In[211]:


y_pred=classifier_svm_sigmoid.predict(x_val)


# In[212]:


y_pred


# In[213]:


y_pred=classifier_dt.predict(x_val)


# In[214]:


y_pred


# In[215]:


y_pred=classifier_rf.predict(x_val)


# In[216]:


y_pred


# In[ ]:




