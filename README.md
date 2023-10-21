# EX-06 FEATURE TRANSFORMATION
## Aim:
To read the given data and perform Feature Transformation process and save the data to a file.

## Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## Algorithm:
### Step1:
Read the given Data.
### Step2: 
Clean the Data Set using Data Cleaning Process.
### Step3:
Apply Feature Transformation techniques to all the features of the data set.
### Step4:
Print the transformed features.
### Program:
Developed By: Yogeshvar M              
Register No: 212222230180

### Importing libraries and reading csv file:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
df=pd.read_csv("Data_to_Transform.csv")
```
### Basic Information:
```
df.head()
df.info()
df.info()
```
![image](https://github.com/Yogeshvar005/ODD2023-Datascience-Ex06/assets/113497367/2aafe8ea-be5b-4712-bb52-8f2f1c113f25)
  
### Before Transformation:
```
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/Yogeshvar005/ODD2023-Datascience-Ex06/assets/113497367/4e6ce3a2-5b3b-4fd1-9180-e59484bd6c2a)

```
sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.title("Highly Negative Skew")
plt.show()
```
![image](https://github.com/Yogeshvar005/ODD2023-Datascience-Ex06/assets/113497367/51d50878-3fe9-43ed-8de1-809f7c49bdb5)

```
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```
![image](https://github.com/Yogeshvar005/ODD2023-Datascience-Ex06/assets/113497367/e6aca28d-54a6-4ee6-a3f0-caf876fe6aad)

```
sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
![image](https://github.com/Yogeshvar005/ODD2023-Datascience-Ex06/assets/113497367/69a2ec67-eda1-4174-a634-322c52c656e9)

### Log Transformation:
```
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/Yogeshvar005/ODD2023-Datascience-Ex06/assets/113497367/89a86c23-37f7-4ace-986c-17ec053911e8)

```
df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```
![image](https://github.com/Yogeshvar005/ODD2023-Datascience-Ex06/assets/113497367/d550f176-da92-4925-ac1e-e3fe884ee92e)

### Reciprocal Transformation:
```
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/Yogeshvar005/ODD2023-Datascience-Ex06/assets/113497367/9eca7f11-49e0-44c8-860b-de548b8e1c83)

### SquareRoot Transformation:
```
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/Yogeshvar005/ODD2023-Datascience-Ex06/assets/113497367/36f32fe6-f31f-4e94-b1d9-ee77028d8dcb)

### Power Transformation:
```
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```
![image](https://github.com/Yogeshvar005/ODD2023-Datascience-Ex06/assets/113497367/b94c95cf-e97e-474b-b9fd-1744e6362386)

```
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
![image](https://github.com/Yogeshvar005/ODD2023-Datascience-Ex06/assets/113497367/8d534e66-5149-4d16-9b8f-3aea5cd03ba4)
 
### Quantile Transformation:
```
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate  Negative Skew")
plt.show()
```
![image](https://github.com/Yogeshvar005/ODD2023-Datascience-Ex06/assets/113497367/c4e30d1c-656a-4348-9090-b48c5e3ef7e9)

## Result:
Thus feature transformation is done for the given dataset.
