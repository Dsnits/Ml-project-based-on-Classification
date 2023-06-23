#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[20]:


import numpy as np                       
import pandas as pd                      
import matplotlib.pyplot as plt          
import seaborn as sb                     
import warnings
warnings.filterwarnings('ignore')        
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold   
from sklearn import preprocessing                      
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder      
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score       
from sklearn.metrics import recall_score,f1_score,roc_auc_score,roc_curve    


# In[6]:


df=pd.read_csv("DS3_C6_S2_Classification_HouseGrade_Data_Project.csv")
df


# # Data Preprocessing 

# In[8]:


df.shape


# In[9]:


df.head()


# In[10]:


df.tail()


# # NULL VALUE TREATMENT

# In[15]:


df.isnull().sum()


# In[ ]:


#No null values present in the data set 


# In[14]:


df.describe().T


# In[16]:


df.info()


# In[ ]:


#roof and grade are having object data types only 


# # VISULISATION 

# In[17]:


plt.subplots(1,2,figsize=(12,10))

plt.subplot(121)
explode = [0.1,0]
perc = df['roof'].value_counts()
lab = list(df['roof'].value_counts().index)
plt.pie(perc,labels = lab, autopct= "%0.2f%%", explode = explode, shadow=True)
plt.title('Roof', size=15, fontweight='bold')

plt.subplot(122)
explode = [0.1,0,0,0,0.2]
perc = df['Grade'].value_counts()
lab = list(df['Grade'].value_counts().index)
plt.pie(perc,labels = lab, autopct= "%0.2f%%", explode=explode, shadow=True)
plt.title('Grade', size=15, fontweight='bold')


# # Interpretation:
#     
# - There are 51.43% of houses which have roof and 48.57% dont have roof of their house.
# - In D grade 42.33% have maximum percentage of share in houses and E have the least 2.53%.

# In[23]:


# plt.subplots(2,2,figsize=(14,11))
plt.subplot(221)
plt.title('Total Floors', size=15, fontweight='bold')
plt.xlabel('Total Floors')
sb.histplot(df['Nfloors'], bins=20, kde=True)

plt.subplot(222)
plt.title('Roof Area', size=15, fontweight='bold')
plt.xlabel('Roof Area')
sb.kdeplot(df['Roof(Area)'], shade=True, color='blue')

plt.subplot(223)
plt.title('Expected price', size=15, fontweight='bold')
plt.xlabel('Expected price')
sb.kdeplot(df['Expected price'], shade=True, color='blue')

plt.subplot(224)
plt.title('Total Area', size=15, fontweight='bold')
plt.xlabel('Total Area')
sb.histplot(df['Area(total)'], bins=20, kde=True)


# #Interpertations:
#     1.Max count is having 3 number of floors 
#     2.In Expected Price Graph, the graph  normally distributed where the most expected price are having 3500-4000
#     3.In Total Area Graph, most of the total area is from 295 to 355 having count from 130

# In[24]:


plt.subplots(2,2,figsize=(14,11))

plt.subplot(221)
plt.title('Bedrooms - Total Rooms', size=15, fontweight='bold')
plt.xlabel('Total Rooms')
plt.ylabel('Bedrooms')
sb.boxplot(y=df['Nbedrooms'],x=df['Trooms'])

plt.subplot(222)
plt.title('Total Rooms - Expected Price', size=15, fontweight='bold')
plt.xlabel('Total Rooms')
plt.ylabel('Expected Price')
sb.boxplot(y=df['Expected price'],x=df['Trooms'])

plt.subplot(223)
plt.title('No. Of Floors', size=15, fontweight='bold')
plt.xlabel('No. of Floors')
sb.countplot(x=df['Nfloors'])

plt.subplot(224)
plt.title('Total number of rooms', size=15, fontweight='bold')
plt.xlabel('No. of Rooms')
sb.countplot(x=df['Trooms'], hue=df['Grade'])


# Interpertations: 
# from bedroom vs total rooms graph 
# for total  5 number of rooms-2-4 are bedroom
# for total  6 number of rooms-3-5 are bedroom
# for total  7 number of room-4-6 are bedroom
# for total  8 number of room-5-7 are bedroom
# for total  9 number of room-6-8 are bedroom
# from total rooms vs expected ,For 9 rooms the price is 4200-4700.
# From Nfloors graph , the max count belongs to the 3 floors 
# From Total no of rooms  graph, the max count belongs to GRADE D 
# 

# # conversion of object data types

# In[26]:


col = ['roof','Grade']
label_encoder = LabelEncoder()
for i in col:
    df[i] = df[i].astype('category')                  
    df[i] = label_encoder.fit_transform(df[i])   


# # Data Scalling 

# In[27]:


a = df.drop(['Grade'],axis=1)     


# In[28]:


scaler = MinMaxScaler()                           
df_scaled = pd.DataFrame(scaler.fit_transform(a.to_numpy()),columns=a.columns)  
df_scaled.head()


# # correlation

# In[29]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[30]:


import seaborn as sns
sns.heatmap(corr)
plt.show()


# In[31]:


corr = abs(df.corr())
corr['Grade'].sort_values(ascending=False)


# In[ ]:


# above sorted value to select parameter easily


# # feature selection

# In[32]:


col = df_scaled[['Trooms','Nbedrooms','Nbwashrooms','Twashrooms','Nfloors','Expected price','Area(total)','Roof(Area)','roof']]
df_scaled = col


# #  using StratifiedKFold

# In[41]:


X = np.array(df_scaled)
Y = np.array(df['Grade'])

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, train_size=0.80, random_state=20)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# # MODEL BUILDING - without cross validation

# # NAVIE BAYES 

# In[38]:


from sklearn.naive_bayes import GaussianNB


# In[43]:


nb_model1 = GaussianNB().fit(X_train, Y_train)             
Y_pred_nb1 = nb_model1.predict(X_test)                    
print('Prediction Response :',Y_pred_nb1)


# In[45]:


plt.figure(figsize=(8,6))
sb.heatmap(confusion_matrix(Y_test, Y_pred_nb1), annot=True, fmt='.3g')           
nb_acc1 = round(accuracy_score(Y_test, Y_pred_nb1)*100,3)                         
nb_pre1 = round(precision_score(y_test, Y_pred_nb1, average='weighted')*100,3)    
nb_rec1 = round(recall_score(Y_test, Y_pred_nb1, average='weighted')*100,3)       
nb_f11 = round(f1_score(y_test, Y_pred_nb1, average='weighted')*100,3)            
print('Accuracy Score :',nb_acc1)
print('Precision Score :',nb_pre1)
print('Recall Score :',nb_rec1)
print('F1 Score :',nb_f11)


# In[46]:


Y


# # 2. Naive Bayes - With Cross Validation
# 

# In[47]:


from sklearn.model_selection import cross_validate


# In[48]:


nb_model2 = GaussianNB()
scoring=['accuracy','precision_weighted','recall_weighted','f1_weighted']


nb_scores2 = cross_validate(nb_model2, x, y, scoring=scoring, cv=rskf, n_jobs=-1, error_score='raise')


# In[49]:


nb_acc2 = round(np.mean(nb_scores2['test_accuracy'])*100,3)             
nb_pre2 = round(np.mean(nb_scores2['test_precision_weighted'])*100,3)   
nb_rec2 = round(np.mean(nb_scores2['test_recall_weighted'])*100,3)      
nb_f12 = round(np.mean(nb_scores2['test_f1_weighted'])*100,3)           
print('Accuracy Score :', nb_acc2)
print('Precision Score :', nb_pre2)
print('Recall Score:', nb_rec2) 
print('F1 Score :', nb_f12) 


# ## 3. Decision Tree Classification - Without Cross Validation

# In[50]:


from sklearn.tree import DecisionTreeClassifier


# In[51]:


import sklearn
sklearn.metrics.SCORERS.keys()


# In[53]:


dtc_model1 = DecisionTreeClassifier(max_depth=5).fit(X_train, Y_train)      
Y_pred_dtc1 = dtc_model1.predict(X_test)                                   
print('Predicted Response :',Y_pred_dtc1)


# In[55]:


plt.figure(figsize=(8,6))
sb.heatmap(confusion_matrix(Y_test, Y_pred_dtc1), annot=True, fmt='.3g')            
dtc_acc1 = round(accuracy_score(y_test, Y_pred_dtc1)*100,3)                         
dtc_pre1 = round(precision_score(y_test, Y_pred_dtc1, average='weighted')*100,3)    
dtc_rec1 = round(recall_score(Y_test, Y_pred_dtc1, average='weighted')*100,3)       
dtc_f11 = round(f1_score(Y_test, Y_pred_dtc1, average='weighted')*100,3)            
print('Accuracy Score :',dtc_acc1)
print('Precision Score :',dtc_pre1)
print('Recall Score :',dtc_rec1)
print('F1 Score :',dtc_f11)


# # 4. Decision Tree Classifier - With Cross Validation

# In[57]:


dtc_model2 = DecisionTreeClassifier()
scoring=['accuracy','precision_weighted','recall_weighted','f1_weighted']


dtc_scores2 = cross_validate(dtc_model2, X, Y, scoring=scoring, cv=rskf, n_jobs=-1, error_score='raise')


# In[58]:


dtc_acc2 = round(np.mean(dtc_scores2['test_accuracy'])*100,3)               
dtc_pre2 = round(np.mean(dtc_scores2['test_precision_weighted'])*100,3)     
dtc_rec2 = round(np.mean(dtc_scores2['test_recall_weighted'])*100,3)        
dtc_f12 = round(np.mean(dtc_scores2['test_f1_weighted'])*100,3)             
print('Accuracy Score :', dtc_acc2)
print('Precision Score :', dtc_pre2)
print('Recall Score:', dtc_rec2) 
print('F1 Score :', dtc_f12) 


# # 5. Random Forest Classifier - Without Cross Validation

# In[60]:


from sklearn.ensemble import RandomForestClassifier


# In[61]:


rfc_model1 = RandomForestClassifier(max_depth=5).fit(X_train, Y_train)    
Y_pred_rfc1 = rfc_model1.predict(X_test)                                  
print('Predicted response :',Y_pred_rfc1)


# In[63]:


plt.figure(figsize=(8,6))
sb.heatmap(confusion_matrix(Y_test, Y_pred_rfc1), annot=True, fmt='.3g')           
rfc_acc1 = round(accuracy_score(Y_test, Y_pred_rfc1)*100,3)                        
rfc_pre1 = round(precision_score(Y_test, Y_pred_rfc1, average='weighted')*100,3)   
rfc_rec1 = round(recall_score(y_test, Y_pred_rfc1, average='weighted')*100,3)      
rfc_f11 = round(f1_score(Y_test, Y_pred_rfc1, average='weighted')*100,3)           
print('Accuracy Score :',rfc_acc1)
print('Precision Score :',rfc_pre1)
print('Recall Score :',rfc_rec1)
print('F1 Score :',rfc_f11)


# # 6. Random Forest Classifier - With Cross Validation

# In[64]:


rfc_model2 = RandomForestClassifier()
scoring = ['accuracy','precision_weighted','recall_weighted','f1_weighted']

rfc_scores2 = cross_validate(rfc_model2, X,Y, cv=rskf, n_jobs=-1, scoring=scoring, error_score='raise')


# In[65]:


rfc_acc2 = round(np.mean(rfc_scores2['test_accuracy'])*100,3)               
rfc_pre2 = round(np.mean(rfc_scores2['test_precision_weighted'])*100,3)     
rfc_rec2 = round(np.mean(rfc_scores2['test_recall_weighted'])*100,3)        
rfc_f12 = round(np.mean(rfc_scores2['test_f1_weighted'])*100,3)             
print('Accuracy Score :', rfc_acc2)
print('Precision Score :', rfc_pre2)
print('Recall Score:', rfc_rec2) 
print('F1 Score :', rfc_f12)


# In[2]:


compare = pd.DataFrame({'Accuracy Score':[nb_acc1,nb_acc2,dtc_acc1,dtc_acc2,rfc_acc1,rfc_acc2,],
                        'Precision Score':[nb_pre1,nb_pre2,dtc_pre1,dtc_pre2,rfc_pre1,rfc_pre2],
                        'Recall Score':[nb_rec1,nb_rec2,dtc_rec1,dtc_rec2,rfc_rec1,rfc_rec2],
                        'F1 Score':[nb_f11,nb_f12,dtc_f11,dtc_f12,rfc_f11,rfc_f12]},                       
                      index=['Naive Bayes without CV','Naive Bayes with CV','Decision Tree without CV',
                             'Decision Tree with CV','Random Forest without CV','Random Forest with CV'])
compare


# # Best model selection 

# # As Per above 4 model the best one is Random Forest with cv is best among all the precision ,accuracy and f1 score all are comperatively high

# # SUGGESTIONS

# # 1.In Grade and Expected price graph, Grade A have highest Expected price from 3600 to more than 5000 and Grade D and E have price range from 2500 to 4500. Most of the count is of Grade C and D.
# 
# 

# # In Floors and Expected Price graph, 6 and 7 Floor building have highest price and least for 1,2,3 and 4 floor building. Most of the building have floors of 3,4 and 5.

# # On the basis of above business implication buyer can decide the type of he/she wants to buy the house.

# In[ ]:




