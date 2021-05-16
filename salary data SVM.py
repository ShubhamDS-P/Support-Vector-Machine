# -*- coding: utf-8 -*-
"""
Created on Sat May 15 20:56:08 2021

@author: Shubham
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Importing the data
# In this project we are already being provided with the train and test data so we don't have to use the 
# train_test_split function on the data 
# importing the train data
salary_train = pd.read_csv("D:\\Data Science study\\Assignment of Data Science\\Sent\\16 Support Vector Machine\\SalaryData_Train(1).csv")

#Importing the test data
salary_test = pd.read_csv("D:\\Data Science study\\Assignment of Data Science\\Sent\\16 Support Vector Machine\\SalaryData_Test(1).csv")

#see the data 
salary_train.describe()
salary_train.info()
salary_train.shape
#columns
salary_train.columns

# Let's copy the data so we can edit it without interfering with the original dataframe
#salary_train_needed = salary_train.copy()

# Deleting un-necessary columns
salary_train_needed = salary_train.drop(["educationno","maritalstatus","relationship","race","sex","native"],axis = 1)
salary_train_needed
salary_train_needed.columns
# we will sea if there are any outlier behaviour present in the numerical data 
sb.boxplot(data = salary_train_needed)

# we will view them seperately to get more cleat view
sb.boxplot(data = salary_train_needed.age)

#There are few outliers can be seen in the graph so we will try to detect them using the funtion below

outliers=[]
def detect_outlier(data_1):         # Creating a function for finding the outliers
    outliers.clear()
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outliers = detect_outlier(salary_train_needed.age)
print(outliers)    #printing the outliers
len(outliers)  #   120    calculating the number of outliers present in the string at the moment


z = np.array(outliers)    #can't apply unique on the list directly 
np.unique(outliers)
len(np.unique(outliers))  # 11 unique values are in the outliers

# Outliers in the capital gain column
sb.boxplot(data = salary_train_needed.capitalgain)
sb.boxplot(data = salary_train_needed.hoursperweek)
# in this plot for hoursperweek there are also lots of outliers present so lets see what are those outliers
#and how many of them are present in our data.
outliers = detect_outlier(salary_train_needed.hoursperweek)
print(outliers)    #printing the outliers
len(outliers)  #   924 calculating the number of outliers present in the string at the moment
#lets see how many unique values are there in this list
z = np.array(outliers)    #can't apply unique on the list directly 
np.unique(outliers)
len(np.unique(outliers))   # 24 unique values of outliers are present in this column

# we will see the count plot for these columns to get their general nature of the data
salary_train_needed.columns
sb.countplot(x = 'age', data = salary_train_needed, palette = 'hls'); plt.xticks(rotation = 90)
sb.countplot(x = 'workclass', data = salary_train_needed, palette = 'hls'); plt.xticks(rotation = 45)
sb.countplot(x = 'education', data = salary_train_needed, palette = 'hls'); plt.xticks(rotation = 45)
sb.countplot(x = 'occupation', data = salary_train_needed, palette = 'hls'); plt.xticks(rotation = 25)
sb.countplot(x = 'capitalgain', data = salary_train_needed, palette = 'hls'); plt.xticks(rotation = 90)
sb.countplot(x = 'capitalloss', data = salary_train_needed, palette = 'hls'); plt.xticks(rotation = 90)
sb.countplot(x = 'hoursperweek', data = salary_train_needed, palette = 'hls'); plt.xticks(rotation = 90)

# Now we will plot the bar graphs for the few of the columns to see their performance and behaviour against the 
#desired output of our model

pd.crosstab(salary_train_needed.age,salary_train_needed.Salary).plot(kind = "bar")
pd.crosstab(salary_train_needed.workclass,salary_train_needed.Salary).plot(kind = "bar")
pd.crosstab(salary_train_needed.education,salary_train_needed.Salary).plot(kind = "bar")
pd.crosstab(salary_train_needed.occupation,salary_train_needed.Salary).plot(kind = "bar")
pd.crosstab(salary_train_needed.hoursperweek,salary_train_needed.Salary).plot(kind = "bar")


salary_test_needed = salary_test.drop(["educationno","maritalstatus","relationship","race","sex","native"],axis = 1)

#so we have created the new data frame where we have taken only the required data for the model building.

# Now we will split the input and output data for further process

output_data_train = salary_train_needed.iloc[:,7]
output_data_test = salary_test_needed.iloc[:,7]
input_data_train = salary_train_needed.iloc[:,:7]
input_data_test = salary_test_needed.iloc[:,:7]

input_data_train.columns

#We will create dummy variables for the catagorical data in the dataframe

model_data_train = pd.get_dummies(data = input_data_train, columns = ['workclass', 'education', 'occupation'])
model_data_test = pd.get_dummies(data = input_data_test, columns = ['workclass', 'education', 'occupation'])
model_data_train


from sklearn.svm import SVC

# There are different different types of kernels in the SVM for us to use.
# These kernel produce different outputs as well so we will be trying them all and see which 
#  kernel gives us the best accuracy and results that we desire.

# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# These are the kernels available for the SVM to use, so let's see them one by one
-------------------------------------------------------------------------------------------------------------------------
# Kernel 'Linear'
model_linear = SVC(kernel = 'linear')
model_linear.fit(model_data_train,output_data_train)
predict_linear_test = model_linear.predict(model_data_test)

# Accuracy
accuracy = np.mean(predict_linear_test = output_data_test)
accuracy   # %         for the linear kernel the model keeps running forever and so we will try other kernels instead.
-------------------------------------------------------------------------------------
# Kernel 'poly'
model_poly = SVC(kernel = 'poly')
model_poly.fit(model_data_train,output_data_train)
predict_poly_test = model_poly.predict(model_data_test)

# Accuracy
accuracy = np.mean(predict_poly_test = output_data_test)
accuracy   # %         for the poly kernel the model keeps running forever and so we will try other kernels instead.
-----------------------------------------------------------------------------------------------------------------------------------
# Kernel 'rbf'
model_rbf = SVC(kernel = 'rbf')
model_rbf.fit(model_data_train,output_data_train)
predict_rbf_test = model_rbf.predict(model_data_test)

# Accuracy
accuracy = np.mean(predict_rbf_test == output_data_test)
accuracy   # 83.51925%
-------------------------------------------------------------------------------
# Kernel 'sigmoid'
model_sigmoid = SVC(kernel = 'sigmoid')
model_sigmoid.fit(model_data_train,output_data_train)
predict_sigmoid_test = model_sigmoid.predict(model_data_test)

# Accuracy
accuracy = np.mean(predict_sigmoid_test == output_data_test)
accuracy   # 75.43160%
----------------------------------------------------------------------------------

# Of these results we have got the best accuracy of 83.51% from the rbf kernel.