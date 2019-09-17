#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


train = pd.read_csv('D:/Competition/Loan Prediction - Analytics Vidhya/train_ctrUa4K.csv')
test = pd.read_csv('D:/Competition/Loan Prediction - Analytics Vidhya/test_lAUu6dG.csv')


# In[108]:


train.columns


# In[109]:


test_orig = pd.read_csv('D:/Competition/Loan Prediction - Analytics Vidhya/test_lAUu6dG.csv')


# In[6]:


test.columns


# In[7]:


train.info()


# In[10]:


print(train.shape)
print(test.shape)


# In[11]:


train['Loan_Status'].value_counts()


# In[12]:


train['Loan_Status'].value_counts(normalize=True)


# In[15]:


train['Loan_Status'].value_counts().plot.bar()


# In[20]:


train['Gender'].value_counts(normalize=True).plot.bar() 


# In[21]:


train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 


# In[22]:


train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 


# In[26]:


train['Dependents'].value_counts(normalize=True).plot.bar()


# In[28]:


train['Education'].value_counts(normalize=True).plot.barh()


# In[29]:


sns.distplot(train['ApplicantIncome'])


# In[30]:


train['ApplicantIncome'].plot.box(figsize=(16,5)) 


# In[33]:


train.boxplot(column='ApplicantIncome', by = 'Education')
plt.suptitle("")


# In[39]:


plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome'])
plt.subplot(122) 
train['CoapplicantIncome'].plot.box(figsize=(16,5)) 
plt.show()


# In[40]:


plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(df['LoanAmount'])
plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5)) 
plt.show()


# In[41]:


Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# In[42]:


Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status']) 
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.show()


# In[44]:


train.isnull().sum()


# In[45]:


train['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)


# In[46]:


train['Loan_Amount_Term'].value_counts()


# In[47]:


train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)


# In[48]:


train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# In[49]:


train.isnull().sum()


# In[50]:


test['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True) 
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# In[52]:


train['LoanAmount_log'] = np.log(train['LoanAmount']) 
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log'] = np.log(test['LoanAmount'])


# In[104]:


df = train.copy()
df2 = test.copy()


# In[54]:


train=train.drop('Loan_ID',axis=1) 
test=test.drop('Loan_ID',axis=1)


# In[55]:


X = train.drop('Loan_Status',1) 
y = train.Loan_Status


# In[56]:


X=pd.get_dummies(X) 
train=pd.get_dummies(train) 
test=pd.get_dummies(test)


# In[85]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size =0.31)


# In[86]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
model = LogisticRegression() 
model.fit(x_train, y_train)


# In[87]:


pred = model.predict(x_test)


# In[88]:


accuracy_score(pred,y_test)


# In[124]:


pred_test = model.predict(test)


# In[125]:


submission = pd.DataFrame({
        "Loan_ID": test_orig["Loan_ID"],
        "Loan_Status": pred_test
    })


# In[126]:


#submission['Loan_Status'] = submission['Loan_Status'].map({'Y': 1, 'N': 0})


# In[127]:


submission


# In[128]:


submission.to_csv('D:/Competition/Loan Prediction - Analytics Vidhya/logistic.csv', index=False)
print('Exported')


# In[ ]:





# In[150]:


from sklearn.model_selection import StratifiedKFold


# In[155]:


i=1 
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y): 
    print('\n{} of kfold {}'.format(i,kf.n_splits)) 
    xtr,xvl = X.loc[train_index],X.loc[test_index]  
    ytr,yvl = y[train_index],y[test_index]       
    model2 = LogisticRegression(random_state=1)  
    model2.fit(xtr, ytr)  
    pred_test2 = model2.predict(xvl)  
    score = accuracy_score(yvl,pred_test2) 
    print('accuracy_score',score)   
    i+=1 
    pred_test2 = model2.predict(test) 
    pred2=model2.predict_proba(xvl)[:,1]


# In[156]:


submission = pd.DataFrame({
        "Loan_ID": test_orig["Loan_ID"],
        "Loan_Status": pred_test2
    })
submission.to_csv('D:/Competition/Loan Prediction - Analytics Vidhya/Stratified_KMean.csv', index=False)
print('Exported')


# In[158]:


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome'] 
test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']


# In[159]:


sns.distplot(train['Total_Income'])


# In[160]:


train['Total_Income_log'] = np.log(train['Total_Income']) 
sns.distplot(train['Total_Income_log'])


# In[161]:


test['Total_Income_log'] = np.log(test['Total_Income'])


# In[163]:


train['EMI']=train['LoanAmount']/train['Loan_Amount_Term'] 
test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']


# In[164]:


sns.distplot(train['EMI'])


# In[165]:


train['Balance Income']=train['Total_Income']-(train['EMI']*1000)


# In[166]:


test['Balance Income']=test['Total_Income']-(test['EMI']*1000)


# In[167]:


sns.distplot(train['Balance Income']);


# In[168]:


train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1) 
test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)


# In[ ]:





# In[ ]:




