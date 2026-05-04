<H3>SHARIKA R</H3>
<H3>212223230204</H3>
<H3>EX. NO.1</H3>
<H3>04/05/2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv("Churn_Modelling.csv",index_col="RowNumber")
df.head()

df.isnull().sum()    

df.duplicated().sum()

df.describe()

df = df.drop(['Surname', 'Geography', 'Gender'], axis=1)

scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)

x = df.iloc[:, :-1].values
x

y = df.iloc[:, -1].values
y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```
## OUTPUT:
<img width="1168" height="215" alt="image" src="https://github.com/user-attachments/assets/6c2fb3dc-62a6-4edb-bd60-795bf36a3efb" />

<img width="277" height="417" alt="image" src="https://github.com/user-attachments/assets/e5252ead-bf84-4e65-94e9-79945609d374" />

<img width="1161" height="276" alt="image" src="https://github.com/user-attachments/assets/acfc55a2-c0c6-4481-bb37-16ba65cce73b" />

<img width="1074" height="799" alt="image" src="https://github.com/user-attachments/assets/668aba14-3ed4-48c3-a510-b1d0c742c503" />

<img width="969" height="382" alt="image" src="https://github.com/user-attachments/assets/9383344e-cf76-4384-b7da-6ae75d4f268a" />

<img width="578" height="33" alt="image" src="https://github.com/user-attachments/assets/8ade4a24-2ec9-4774-ad24-c125b0b384f6" />

<img width="834" height="418" alt="image" src="https://github.com/user-attachments/assets/d918adbe-47d1-4a45-81da-896dbadecb21" />

<img width="827" height="413" alt="image" src="https://github.com/user-attachments/assets/26763e79-abb6-4bf7-b90a-86f41504796d" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


