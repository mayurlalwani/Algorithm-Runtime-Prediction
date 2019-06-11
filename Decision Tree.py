#!/usr/bin/env python
# coding: utf-8

# In[45]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


#dataset = pd.read_csv('category_encode_1.csv')
#dataset = pd.read_csv('483.xalancbmk.csv')
#dataset = pd.read_csv('All algorithms/400.perlbench.csv')
dataset = pd.read_csv('400.perlbench_type.csv')


#dataset = pd.read_csv('category_encode_without_fsb.csv')
#dataset = pd.read_csv('Normalized_info_final_1.csv')


# In[48]:


dataset.head()


# In[49]:


dataset.describe()


# In[50]:


X = dataset.drop(['runtime','l1d_cache_lines','l2_cache_lines','l3_cache_lines','num-cpus','system_name'], axis=1)
y = dataset['runtime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)

regressor = DecisionTreeRegressor()  
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(df)


# In[51]:


#FEATURE IMPORTANCE

regressor2 = DecisionTreeRegressor()  
regressor2.fit(X, y)
feature_importance = regressor2.feature_importances_
print(feature_importance)
label=list(X.columns.values)
#print(label)
bar_index=np.arange(len(feature_importance))
plt.bar(bar_index, feature_importance)
plt.xticks(bar_index, label, fontsize=10, rotation=90)
plt.show()


# In[52]:


plt.scatter(y_test,y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()


# In[59]:


#print(y_test.tolist())
test_data = y_test.tolist()
for items in test_data:
    print(items)


# In[60]:


test_error = (y_test - y_pred)/y_test * 100


# In[61]:


print (test_error.shape)
print(np.mean(abs(test_error)))


# In[56]:


plt.boxplot([y_test,y_pred])
plt.show()


# In[329]:


predicted_values = []
for items in y_pred:
    predicted_values.append((float(items)))


# In[330]:


count=0
for i in predicted_values:
    count+=1
    print(count,i)


# In[331]:


actual_values = []
for items in y_test:
    actual_values.append(float(items))
    


# In[332]:


count=0
for j in actual_values:
    count+=1
    print(count,j)


# In[333]:


#Simple Diff
diff = []
sorted_diff = []
for i in range(len(actual_values)):
    diff.append(abs(actual_values[i]-predicted_values[i]))
    sorted_diff.append(abs(actual_values[i]-predicted_values[i]))

sort = sorted(sorted_diff)


# In[334]:


#Percentage Difference
diff = []
sorted_diff = []
for i in range(len(actual_values)):
    perc_diff = abs(actual_values[i]-predicted_values[i]) / actual_values[i]* 100
    diff.append(perc_diff)
    sorted_diff.append(perc_diff)

sort = sorted(sorted_diff)


# In[335]:


counts=0
for i in diff:
    counts+=1
    print(counts,i)


# In[336]:


count=0
for i in sort:
    count+=1
    print(count,i)


# In[337]:


#print(X_test)


# In[338]:


plt.boxplot(diff)
plt.show()


# In[339]:


print(type(y_test))


# In[340]:


plt.boxplot(diff)
plt.show()


# In[343]:


print(y_train)


# In[342]:


import os     
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

export_graphviz(regressor, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[355]:


export_graphviz(regressor, out_file ='tree1.dot')  


# In[ ]:





# In[ ]:




