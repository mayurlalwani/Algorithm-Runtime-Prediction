#!/usr/bin/env python
# coding: utf-8

# In[58]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import json
import os
import fnmatch

get_ipython().run_line_magic('matplotlib', 'inline')


# In[59]:


#If you are running the int benchmark
#benchmark_type = "Int"
#If you are running the float benchmark use this
benchmark_type="Int"

if(benchmark_type=="Int"):
    benchmark_folder = "./Int/"

    
else:
    benchmark_folder = "./Float/"

    


# In[60]:


benchmark_files = []
for root, dirnames, filenames in os.walk(benchmark_folder):
    for filename in fnmatch.filter(filenames, '4*.csv'):
        benchmark_files.append(os.path.join(root, filename))
print (benchmark_files)


# In[61]:


learned_models = {}
for benchmark_file in benchmark_files:
    # Create an empty dictionary for algo
    benchmark_file_s = benchmark_file.split(".")
    print (benchmark_file_s[2])
    learned_models[benchmark_file_s[2]] = {}
    
    dataset = pd.read_csv(benchmark_file)
    print (dataset.head())
    print (dataset.describe())
    
    # For training and testing (performance prediction)
    X = dataset.drop(['system_name', 'arch', 'mem_type', 'runtime', 'bus_speed_qpi', 'bus_speed_dmi', 'raw_bus_speed'], axis=1)
    #X = dataset.drop(['system_name','l1d_cache_lines', 'l2_cache_lines', 'l3_cache_lines', 'runtime'], axis=1)
    y = dataset['runtime']
    cpu_clock_speed=dataset['cpu_clock']
    mem_clock=dataset['mem_clock']
    num_cpus=dataset['num-cpus']
    num_threads=dataset['no_of_threads']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)

    y_test_pred = regressor.predict(X_test)
    y_train_pred = regressor.predict(X_train)
    #df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
    #print(df)

    #train_test_df=pd.DataFrame({'Actual Training':y_train, 'Predicted Training':y_train_pred})
    #print(train_test_df)
    
    # For feature importance
    regressor_feat_imp = DecisionTreeRegressor()  
    regressor_feat_imp.fit(X, y)

    feature_importance = regressor_feat_imp.feature_importances_
    print(feature_importance)
    label=list(X.columns.values)
    print(label)
    bar_index=np.arange(len(feature_importance))
    plt.bar(bar_index, feature_importance)
    plt.xticks(bar_index, label, fontsize=10, rotation=90)
    plt.show()
    
    # Store everything in dictionary
    learned_models[benchmark_file_s[2]]['algo'] = benchmark_file_s[2]
    learned_models[benchmark_file_s[2]]['model_pred'] = regressor
    learned_models[benchmark_file_s[2]]['model_feat_imp'] = regressor_feat_imp.feature_importances_
    learned_models[benchmark_file_s[2]]['X'] = X
    learned_models[benchmark_file_s[2]]['y'] = y
    learned_models[benchmark_file_s[2]]['X_train'] = X_train
    learned_models[benchmark_file_s[2]]['X_test'] = X_test
    learned_models[benchmark_file_s[2]]['y_train'] = y_train
    learned_models[benchmark_file_s[2]]['y_test'] = y_test
    learned_models[benchmark_file_s[2]]['y_train_pred'] = y_train_pred
    learned_models[benchmark_file_s[2]]['y_test_pred'] = y_test_pred
    test_error=(y_test - y_test_pred)/y_test * 100
    train_error=(y_train - y_train_pred)/y_train * 100
    test_mean_error=np.mean(abs(test_error))
    train_mean_error=np.mean(abs(train_error))
    learned_models[benchmark_file_s[2]]['test_error'] = test_mean_error
    learned_models[benchmark_file_s[2]]['train_error'] = train_mean_error
print (learned_models)


# In[62]:


boxdata_train = []
boxdata_test = []
algos = []
#for learned_model in learned_models.items():
for benchmark_file in benchmark_files:
    # Create an empty dictionary for algo
    benchmark_file_s = benchmark_file.split(".")
    print (benchmark_file_s[2])
    #print "_____________________________________________________________________________________"
    #print learned_model 
    train_error = np.array([])
    test_error = np.array([])
    train_error = np.append(train_error, np.abs((learned_models[benchmark_file_s[2]]['y_train'] - learned_models[benchmark_file_s[2]]['y_train_pred']) / learned_models[benchmark_file_s[2]]['y_train'] * 100))
    test_error = np.append(test_error, np.abs((learned_models[benchmark_file_s[2]]['y_test'] - learned_models[benchmark_file_s[2]]['y_test_pred']) / learned_models[benchmark_file_s[2]]['y_test'] * 100))
    algos.append(learned_models[benchmark_file_s[2]]['algo'])
    boxdata_train.append(train_error)
    boxdata_test.append(test_error)
print (algos)
print (boxdata_train)
print (boxdata_test)

fig = plt.figure(figsize=(len(algos)*.75, 9))
plt.boxplot(boxdata_train, showfliers=False)
plt.xticks(range(1, len(algos)+1), algos, rotation=90)
plt.xlabel("Boxplot for Training Errors across all Algorithms")
plt.show()
strFile = benchmark_folder +"boxplot_train_allalgo_speccpu2006_"+benchmark_type+".png"
if os.path.isfile(strFile):
    print (strFile)
    os.system("del "+strFile)
fig.savefig(strFile)

fig = plt.figure(figsize=(len(algos)*.75, 9))
plt.boxplot(boxdata_test, showfliers=False)
plt.xticks(range(1, len(algos)+1), algos, rotation=90)
plt.xlabel("Boxplot for Testing Errors across all Algorithms")
plt.show()
strFile = benchmark_folder +"boxplot_test_allalgo_speccpu2006_"+benchmark_type+".png"
if os.path.isfile(strFile):
    print (strFile)
    os.system("del "+strFile)
fig.savefig(strFile)


# In[63]:


feat_imp_bar=[]
fi_algo_name=[]
for benchmark_file in benchmark_files:
    # Create an empty dictionary for algo
    benchmark_file_s = benchmark_file.split(".")
    #print "_____________________________________________________________________________________"
    #print learned_model 
    feat_imp_bar.append(learned_models[benchmark_file_s[2]]['model_feat_imp'])
    fi_algo_name.append(learned_models[benchmark_file_s[2]]['algo'])
    #print (feat_imp_bar)

feat_per_graph=4
ind = np.arange(feat_per_graph)
width=0.07
fig, axes = plt.subplots(2, 2, figsize=((feat_per_graph*4), 14))
#plt.subplots(figsize=((feat_per_graph*4), 14))
colors=['r', 'g', 'y', 'gray', 'b', 'orange', 'c', 'chartreuse', 'magenta', 'brown', 'cyan', 'k']
features= learned_models[benchmark_file_s[2]]['X'].columns
#plt.bar(ind, feat_imp_bar[0][0:7], width, label=fi_algo_name[0], color=colors[0])

rects1 = []
rects2 = []
rects3 = []
rects4 = []
for i in range (0, len(fi_algo_name)):
    rects1.append(axes[0, 0].bar(ind + (width*i), feat_imp_bar[i][0:feat_per_graph], width, label=fi_algo_name[i], color=colors[i]))
    axes[0, 0].set_xticks(ind + width * feat_per_graph)
    axes[0, 0].set_xticklabels(features[0:feat_per_graph])
    axes[0, 0].set_ylabel('Importance')
    axes[0, 0].set_xlabel('Features')
    lgd = axes[0, 0].legend((rects1), fi_algo_name)
    rects2.append(axes[0, 1].bar(ind + (width*i), feat_imp_bar[i][feat_per_graph:feat_per_graph*2], width, label=fi_algo_name[i], color=colors[i]))
    axes[0, 1].set_xticks(ind + width * feat_per_graph)
    axes[0, 1].set_xticklabels(features[feat_per_graph:feat_per_graph*2])
    axes[0, 1].set_ylabel('Importance')
    axes[0, 1].set_xlabel('Features')
    lgd = axes[0, 1].legend((rects2), fi_algo_name)
    rects3.append(axes[1, 0].bar(ind + (width*i), feat_imp_bar[i][feat_per_graph*2:feat_per_graph*3], width, label=fi_algo_name[i], color=colors[i]))
    axes[1, 0].set_xticks(ind + width * feat_per_graph)
    axes[1, 0].set_xticklabels(features[feat_per_graph*2:feat_per_graph*3])
    axes[1, 0].set_ylabel('Importance')
    axes[1, 0].set_xlabel('Features')
    lgd = axes[1, 0].legend((rects3), fi_algo_name)
    rects4.append(axes[1, 1].bar(ind + (width*i), feat_imp_bar[i][feat_per_graph*3:feat_per_graph*4], width, label=fi_algo_name[i], color=colors[i]))
    axes[1, 1].set_xticks(ind + width * feat_per_graph)
    axes[1, 1].set_xticklabels(features[feat_per_graph*3:feat_per_graph*4])
    axes[1, 1].set_ylabel('Importance')
    axes[1, 1].set_xlabel('Features')
    lgd = axes[1, 1].legend((rects4), fi_algo_name)
plt.show()
strFile = benchmark_folder +"feature_importance_speccpu2006_"+benchmark_type+".png"
if os.path.isfile(strFile):
    print (strFile)
    os.system("del "+strFile)
fig.savefig(strFile)


# In[64]:


train_errors=[]
test_errors=[]
fi_algo_name=[]
for benchmark_file in benchmark_files:
    # Create an empty dictionary for algo
    benchmark_file_s = benchmark_file.split(".")
    #print "_____________________________________________________________________________________"
    #print learned_model 
    train_errors.append(learned_models[benchmark_file_s[2]]['train_error'])
    test_errors.append(learned_models[benchmark_file_s[2]]['test_error'])
    fi_algo_name.append(learned_models[benchmark_file_s[2]]['algo'])
print(train_errors)
plt.plot(fi_algo_name,train_errors, linestyle='-', marker='o', color='g', label='Train Error')
plt.plot(fi_algo_name,test_errors, linestyle='-', marker='o', color='b', label='Test Error')
plt.xticks(rotation=90)
plt.legend(loc='upper left')
plt.ylim(0, 15)
plt.show()


# In[65]:


dataset.head()


# In[66]:


dataset.describe()


# In[67]:


#training and testing
X = dataset.drop(['system_name', 'arch', 'mem_type', 'runtime', 'bus_speed_qpi', 'bus_speed_dmi', 'raw_bus_speed'], axis=1)
#X = dataset.drop(['system_name','l1d_cache_lines', 'l2_cache_lines', 'l3_cache_lines', 'runtime'], axis=1)
y = dataset['runtime']
cpu_clock_speed=dataset['cpu_clock']
mem_clock=dataset['mem_clock']
num_cpus=dataset['num-cpus']
num_threads=dataset['no_of_threads']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=0)
regressor = DecisionTreeRegressor()  
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_train_pred = regressor.predict(X_train)
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
#print(df)

train_test_df=pd.DataFrame({'Actual Training':y_train, 'Predicted Training':y_train_pred})
#print(train_test_df)


# In[68]:


regressor2 = DecisionTreeRegressor()  
regressor2.fit(X, y)

feature_importance = regressor2.feature_importances_
print(feature_importance)
label=list(X.columns.values)
print(label)
bar_index=np.arange(len(feature_importance))
plt.bar(bar_index, feature_importance)
plt.xticks(bar_index, label, fontsize=10, rotation=90)
plt.show()


# In[69]:


#Analysis
plt.scatter(y_test,y_pred)
plt.xlabel("Test Values")
plt.ylabel("Test Predictions")
plt.show()

plt.scatter(y_train,y_train_pred)
plt.xlabel("Training Values")
plt.ylabel("Training Predictions")
plt.show()


# In[70]:


test_error = (y_test - y_pred)/y_test * 100
train_error = (y_train - y_train_pred)/y_train * 100


# In[71]:


print (test_error.shape)
test_mean_error=np.mean(abs(test_error))
train_mean_error=np.mean(abs(train_error))
print("Test mean error",test_mean_error)
print("Train mean error",train_mean_error)


# In[72]:


#plt.boxplot([[test_mean_error],[train_mean_error]])
#plt.show()


# In[73]:


predicted_values = []
for items in y_pred:
    predicted_values.append((float(items)))

predicted_training_values = []
for items in y_train_pred:
    predicted_training_values.append((float(items)))


# In[74]:


actual_values = []
for items in y_test:
    actual_values.append(float(items))
    
actual_training_values = []
for items in y_train:
    actual_training_values.append(float(items))
    


# In[75]:


#Simple Diff
diff = []
sorted_diff = []
for i in range(len(actual_values)):
    diff.append(abs(actual_values[i]-predicted_values[i]))
    sorted_diff.append(abs(actual_values[i]-predicted_values[i]))

sort = sorted(sorted_diff)


# In[33]:


#Percentage Difference
diff = []
sorted_diff = []
for i in range(len(actual_values)):
    perc_diff = abs(((actual_values[i]-predicted_values[i]) / actual_values[i])* 100)
    diff.append(perc_diff)
    sorted_diff.append(perc_diff)

sort = sorted(sorted_diff)


# In[34]:


#Percentage Difference for training
diff_training = []
sorted_training_diff = []
for i in range(len(actual_training_values)):
    perc_diff = abs(((actual_training_values[i]-predicted_training_values[i]) / actual_training_values[i])* 100)
    diff_training.append(perc_diff)
    sorted_training_diff.append(perc_diff)

sort_training = sorted(sorted_training_diff)


# In[35]:


#print(X_test)


# In[36]:


#Boxplot for percentage difference in training and testing 
# data_dump.append(diff_training)
# data_dump.append(diff)
# jobj = {"diff" : data_dump}
# json.dump(jobj, dump_file)
plt.boxplot([diff_training, diff], showfliers=False)
plt.xticks([1, 2], ['Training', 'Testing'])
plt.xlabel("Percentage errors in Training and Testing")
plt.show()


# In[37]:


plt.scatter(y,cpu_clock_speed)
plt.xlabel("Runtime")
plt.ylabel("CPU Clock(in GHz)")
plt.show()


# In[38]:


plt.scatter(y,mem_clock)
plt.xlabel("Runtime")
plt.ylabel("Memory Clock(in MHz)")
plt.show()


# In[39]:


plt.scatter(y,num_cpus)
plt.xlabel("Runtime")
plt.ylabel("Number of CPUs")
plt.show()


# In[40]:


plt.scatter(y,num_threads)
plt.xlabel("Runtime")
plt.ylabel("Number of Threads")
plt.show()


# In[26]:


# plt.boxplot(training, showfliers=False)
# plt.xticks([1, 2, 3, 4], ['80-20', '70-30', '60-40', '50-50'])
# plt.xlabel("Percentage errors in Training")
# plt.show()


# In[44]:


plt.boxplot(training, showfliers=False)
plt.xticks([1, 2, 3, 4], ['80-20', '70-30', '60-40', '50-50'])
plt.xlabel("Percentage errors in Training")
plt.ylabel("Error in percentage")
plt.show()


# In[ ]:


plt.boxplot(testing, showfliers=False)
plt.xticks([1, 2, 3, 4], ['80-20', '70-30', '60-40', '50-50'])
plt.xlabel("Percentage errors in Testing")
plt.ylabel("Error in percentage")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




