# Implementation-of-K-Means-Clustering-for-Customer-Segmentation
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Import dataset and print head,info of the dataset  
2.check for null values   
3.Import kmeans and fit it to the dataset    
4.Plot the graph using elbow method    
5.Print the predicted array    
6.Plot the customer segments    
## Program:
```
Developed by: Surjith D
RegisterNumber:212223043006
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("Mall_Customers.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.xlabel("No_of_Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
km = KMeans(n_clusters=5, init="k-means++", random_state=42)
km.fit(data.iloc[:, 3:])
y_pred = km.predict(data.iloc[:, 3:])
print(y_pred)
data["cluster"] = y_pred
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="cluster 0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="black", label="cluster 1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="blue", label="cluster 2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="green", label="cluster 3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="magenta", label="cluster 4")
plt.legend()
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()
```
## Output:
![WhatsApp Image 2025-05-09 at 09 13 56_0ad3c11c](https://github.com/user-attachments/assets/6fdfee69-50bb-4519-b0d2-c24a3b2c39b9)
![image](https://github.com/user-attachments/assets/079a1e07-d452-4ab0-b429-6035e1304f6b)
![image](https://github.com/user-attachments/assets/f30c5291-b26d-4ec6-aa2c-d0eefec43544)
![WhatsApp Image 2025-05-09 at 09 05 08_9152f425](https://github.com/user-attachments/assets/8482516a-8223-4838-a727-c028aa1ba30c)
## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
