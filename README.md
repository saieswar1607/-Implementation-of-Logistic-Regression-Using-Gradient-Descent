# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.
2. Read the given dataset and assign x and y array.
3. Split x and y into training and test set.
4. Scale the x variables.
5. Fit the logistic regression for the training set to predict y.
6. Create the confusion matrix and find the accuracy score, recall sensitivity and specificity.
7. Plot the training set results.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sai Eswar Kandukuri
RegisterNumber:  212221240020
*/
#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#reading and displaying dataframe
df=pd.read_csv("Social_Network_Ads (1).csv")
df

#assigning x and y and displaying them
x=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values 

#splitting data into train and test
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)

#scaling values and obtaining scaled array of train and test of x
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)

#applying logistic regression to the scaled array
from sklearn.linear_model import LogisticRegression
c=LogisticRegression(random_state=0)
c.fit(xtrain,ytrain)

#finding predicted values of y
ypred=c.predict(xtest)
ypred

#calculating confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
cm

#calculating accuracy score
from sklearn import metrics
acc=metrics.accuracy_score(ytest,ypred)
acc

#calculating recall sensitivity and specificity
r_sens=metrics.recall_score(ytest,ypred,pos_label=1)
r_spec=metrics.recall_score(ytest,ypred,pos_label=0)
r_sens,r_spec

#displaying regression 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
xs,ys=xtrain,ytrain
x1,x2=np.meshgrid(np.arange(start=xs[:,0].min()-1,stop=xs[:,0].max()+1,step=0.01),
               np.arange(start=xs[:,1].min()-1,stop=xs[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,c.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                            alpha=0.75,cmap=ListedColormap(("pink","purple")))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x1.max())
for i,j in enumerate(np.unique(ys)):
    plt.scatter(xs[ys==j,0],xs[ys==j,1],
                c=ListedColormap(("white","violet"))(i),label=j)
plt.title("Logistic Regression(Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
```

## Output:
<img width="665" alt="1" src="https://user-images.githubusercontent.com/93427011/175761356-1596ce96-658e-4bd6-bc74-809da7ffa80e.png">
<img width="830" alt="2" src="https://user-images.githubusercontent.com/93427011/175761359-a41f95f5-06fd-4e47-936e-3e81f2989eea.png">
<img width="586" alt="3" src="https://user-images.githubusercontent.com/93427011/175761366-ad20d3f0-a52d-4874-9e1d-f824571895f9.png">
<img width="382" alt="4" src="https://user-images.githubusercontent.com/93427011/175761368-42733e33-b85d-44a6-8a50-86936f25210c.png">
<img width="575" alt="5" src="https://user-images.githubusercontent.com/93427011/175761374-858f4ca4-86fb-4526-92db-c8b36b282b6d.png">
<img width="741" alt="6" src="https://user-images.githubusercontent.com/93427011/175761375-a6eb650a-1a34-4e50-826a-04e4d4a2a733.png">


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

