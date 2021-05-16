# -*- coding: utf-8 -*-
"""
Created on Sun May 16 12:44:09 2021

@author: Shubham
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

forestfire = pd.read_csv("D:\\Data Science study\\Assignment of Data Science\\Sent\\15 Neural Network\\forestfires.csv")
print(forestfire)

forestfire.head
forestfire.info()
forestfire.columns
forestfire.select_dtypes(object)   # Getting only the object type of column from the dataframe.
# checking if there are any null values in the dataframe
forestfire.isnull().sum()    
# We can say that there are no null values in the dataset

import seaborn as sb

sb.boxplot(forestfire.FFMC)
sb.boxplot(forestfire.DMC)
sb.boxplot(forestfire.DC)
sb.boxplot(forestfire.ISI)
sb.boxplot(forestfire.temp)
sb.boxplot(forestfire.RH)
sb.boxplot(forestfire.wind)

# Let's create a function for detecting the outliers present in the above graphs.

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

#Let's see the outliers now.

detect_outlier(forestfire.FFMC)
len(outliers)    # There are 7 outliers in this column

detect_outlier(forestfire.DMC)
len(outliers)    # There are no outliers in this column

detect_outlier(forestfire.DC)
len(outliers)    # There are no outliers in this column

detect_outlier(forestfire.ISI)
len(outliers)    # There are 2 outliers in this column

detect_outlier(forestfire.temp)
len(outliers)    # There are no outliers in this column

detect_outlier(forestfire.RH)
len(outliers)    # There are 5 outliers in this column

detect_outlier(forestfire.wind)
len(outliers)    # There are 4 outliers in this column

# first let's drop the unnecessary columns from the data frame.

data = forestfire.iloc[:,2:]
print(data)

# Now let's assign the values to the bicategorical output column for the model building

data['size_category'].describe()
data['size_category'].unique()

#Let's assign 0 to the 'small' and 1 to the 'large'

data.loc[data.size_category=='small','size_category'] = 0
data.loc[data.size_category=='large','size_category'] = 1

# Let's create X and Y for the model

x = data.iloc[:,:28]
y = data.iloc[:,28]

plt.hist(y)

data.size_category.value_counts()

# Let's split the data into train and test data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2)

from sklearn.svm import SVC

# There are different different types of kernels in the SVM for us to use.
# These kernel produce different outputs as well so we will be trying them all and see which 
#  kernel gives us the best accuracy and results that we desire.

# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# These are the kernels available for the SVM to use, so let's see them one by one
--------------------------------------------------------------------------------
# Kernel 'Linear'
model_linear = SVC(kernel = 'linear')
model_linear.fit(x_train,y_train)
predict_linear_test = model_linear.predict(x_test)

# Accuracy
accuracy = np.mean(predict_linear_test == y_test)
accuracy   # 99.03846153846154%
-------------------------------------------------------------------------------
# Kernel 'rbf'
model_rbf = SVC(kernel = 'rbf')
model_rbf.fit(x_train,y_train)
predict_rbf_test = model_rbf.predict(x_test)

# Accuracy
accuracy = np.mean(predict_rbf_test == y_test)
accuracy   # 76.92307692307693%
-----------------------------------------------------------------------------------
# Kernel 'poly'
model_poly = SVC(kernel = 'poly')
model_poly.fit(x_train,y_train)
predict_poly_test = model_poly.predict(x_test)

# Accuracy
accuracy = np.mean(predict_poly_test == y_test)
accuracy   # 97.11538461538461%
-------------------------------------------------------------------------------------
# Kernel 'sigmoid'
model_sigmoid = SVC(kernel = 'sigmoid')
model_sigmoid.fit(x_train,y_train)
predict_sigmoid_test = model_sigmoid.predict(x_test)

# Accuracy
accuracy = np.mean(predict_sigmoid_test == y_test)
accuracy   # 74.03846153846154%
-------------------------------------------------------------------------------

#In this so far we got some of the best results in the two kernels which are 'linear' and 'poly' [polynomial].
#Here we can say that 'linear' kernel gives us the best result possible for the prediction of the model based on the given data
#and we can use it for further predictions of our data
#The best result so far is the accuracy of 99.03846153846154%.