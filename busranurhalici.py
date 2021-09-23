# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:50:04 2020

@author: PC
"""
# kütüphanelerin çağrılması
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# verinin yüklenmesi
df= pd.read_csv('student-mat.csv')

# verinin ilk 8 satırının gösterilmesi
print(df.head(8))

# verinin son 8 satırının gösterilmesi
print(df.tail(8))

# verinin kaç satır ve sütundan oluştuğunu gösterme
print(df.shape)

# verideki  özellikler
print(df.columns.values)

# veri seti icindeki ozelliklerin veri tipleri
print (df.info())

# eksik verilerin analizi
print(df.isnull().sum())

# korelasyon grafiği
corr = df.corr()
sns.heatmap(corr,xticklabels=corr.columns,
        yticklabels=corr.columns)

# istenmeyen kolonların silinmesi
df.drop(['address','reason','guardian','traveltime','activities', 'higher','internet', 'nursery','romantic','health','absences'],axis=1,inplace=True)
print (df.head())
print(df.shape)

# sayısal verilere sahip olan sütunların max,min,std gibi istatiksel değerlerini gösterme
print(df.describe())

# df.describle() çıktısına göre G3'ün ortalaması 10.415'dır
# bu ortalama esas alınarak "Durumu" sütunu oluşturulması
df['Durumu'] = np.where(df['G3'] <= 4 , 'Kötü',
  (np.where((df['G3'] <=8) & (df['G3'] > 4) ,'Yeterli',
            (np.where((df['G3'] <= 12) & (df['G3'] >8) , 'Orta',
                      (np.where((df['G3'] <= 16) & (df['G3'] >12), 'İyi','Çok iyi')))))))
print(df.head())
print(df.tail())

# G3 sütununun silinmesi
df.drop(columns="G3",axis=1,inplace=True)

# Öğrencilerin başarı durumunu gösteren grafik
fig=plt.figure(figsize=(5,5))
df["Durumu"].value_counts().plot(kind='pie', autopct= "%.1f%%")
plt.title("Öğrencilerin Başarı Durumları")
plt.legend()
plt.show()

# Durumu kötü olan öğrencilerin cinsiyet dağılımı
df[df["Durumu"]=="Kötü"].groupby("sex").size().plot(kind='bar',color='#B71C1C',edgecolor='black')
plt.xticks(rotation=0)
plt.xlabel("Cinsiyet")
plt.ylabel("Örnek Sayılar")
plt.title("Başarı Durumu Kötü Olan Öğrencilerin Cinsiyet Dağılımı")
plt.show()

# Durumu çok iyi olan öğrencilerin cinsiyet dağılımı
df[df["Durumu"]=="Çok iyi"].groupby("sex").size().plot(kind='bar',color='#4CAF50',edgecolor='black')
plt.xticks(rotation=0)
plt.xlabel("Cinsiyet")
plt.ylabel("Örnek Sayılar")
plt.title("Başarı Durumu Çok İyi Olan Öğrencilerin Cinsiyet Dağılımı")
plt.show()

# Durumu kötü olan öğrencilerin yaş dağılımı
df[df["Durumu"]=="Kötü"].groupby("age").size().plot(kind='bar',color='#B71C1C',edgecolor='black')
plt.xticks(rotation=0)
plt.xlabel("Yaş")
plt.ylabel("Örnek Sayılar")
plt.title("Başarı Durumu Kötü Olan Öğrencilerin Yaş Dağılımı")
plt.grid(True)
plt.show()

# Durumu çok iyi olan öğrencilerin yaş dağılımı
df[df["Durumu"]=="Çok iyi"].groupby("age").size().plot(kind='bar',color='#4CAF50',edgecolor='black')
plt.xticks(rotation=0)
plt.xlabel("Yaş")
plt.ylabel("Örnek Sayılar")
plt.title("Başarı Durumu Çok İyi Olan Öğrencilerin Yaş Dağılımı")
plt.grid(True)
plt.show()

# Failures sayısına göre öğrencilerin başarı durumunu gösteren grafik
b = sns.swarmplot(x=df['failures'],y=df['Durumu'])
b.axes.set_title('Failures Sayısı Düşük Öğrencilerin Başarı Durumu', fontsize = 12)
b.set_xlabel('Failures Sayısı', fontsize = 10)
b.set_ylabel('Başarı Durumu', fontsize = 10)
plt.show()

# Aile eğitimine göre öğrencilerin başarı durumu
family_ed = df['Fedu'] + df['Medu'] 
b = sns.boxplot(x=family_ed,y=df['Durumu'])
b.axes.set_title('Aile Eğitimine Göre Öğrencilerin Başarı Durumu', fontsize = 12)
b.set_xlabel('Aile Eğitimi (Anne ve Baba)', fontsize = 10)
b.set_ylabel('Başarı Durumu', fontsize = 10)
plt.show()


# kategorik değerleri nümerik hale çevirme
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dtype_object=df.select_dtypes(include=['object'])
print (dtype_object.head())
for x in dtype_object.columns:
    df[x]=le.fit_transform(df[x])

print (df.head(15))

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df["Durumu"]=le.fit_transform(df["Durumu"])


# veri setini bağımsız ve bağımlı değişkenlere ayırma
X = df.iloc[:,0:21].values
y = df["Durumu"].values

# veriyi %80 eğitim %20 test olarak bölme
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 55)
print("train:",len(X_train),"test:",len(X_test))

# özellik ölçeklendirme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


score=[]
algorithms=[]

# KNN
from sklearn.neighbors import KNeighborsClassifier

# modeli oluşturma ve eğitim
knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)
knn.predict(X_test)
score.append(knn.score(X_test,y_test)*100)
algorithms.append("KNN")
print("KNN accuracy =",knn.score(X_test,y_test)*100)

# karmaşıklık matrisi
from sklearn.metrics import confusion_matrix
y_pred=knn.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

# korelasyon grafiği
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title(" KNN Confusion Matrix")
plt.show()

from sklearn.metrics import classification_report

target_names=["Kötü","Yeterli","Orta","İyi","Çok iyi"]
print(classification_report(y_true, y_pred, target_names=target_names))


# Naive-Bayes
from sklearn.naive_bayes import GaussianNB

# modeli oluşturma ve eğitim
nb=GaussianNB()
nb.fit(X_train,y_train)
score.append(nb.score(X_test,y_test)*100)
algorithms.append("Naive-Bayes")
print("Naive Bayes accuracy =",nb.score(X_test,y_test)*100)

# karmaşıklık matrisi
from sklearn.metrics import confusion_matrix
y_pred=nb.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

# korelasyon grafiği
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Naive Bayes Confusion Matrix")
plt.show()
target_names=["Kötü","Yeterli","Orta","İyi","Çok iyi"]
print(classification_report(y_true, y_pred, target_names=target_names))


# Support Vector Machine
from sklearn.svm import SVC

# modeli oluşturma ve eğitim
svm=SVC(kernel = 'linear')
svm.fit(X_train,y_train)
score.append(svm.score(X_test,y_test)*100)
algorithms.append("Support Vector Machine")
print("svm test accuracy =",svm.score(X_test,y_test)*100)

# karmaşıklık matrisi
from sklearn.metrics import confusion_matrix
y_pred=svm.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

# korelasyon grafiği
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Support Vector Machine Confusion Matrix")
plt.show()
target_names=["Kötü","Yeterli","Orta","İyi","Çok iyi"]
print(classification_report(y_true, y_pred, target_names=target_names))


# DecisionTree
from sklearn.tree import DecisionTreeClassifier

# modeli oluşturma ve eğitme
dt=DecisionTreeClassifier(criterion = 'entropy', random_state=55)
dt.fit(X_train,y_train)
print("Decision Tree accuracy:",dt.score(X_test,y_test)*100)
score.append(dt.score(X_test,y_test)*100)
algorithms.append("Decision Tree")

# karmaşıklık matrisi
from sklearn.metrics import confusion_matrix
y_pred=dt.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

# korelasyon grafiği
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Decision Tree Confusion Matrix")
plt.show()
target_names=["Kötü","Yeterli","Orta","İyi","Çok iyi"]
print(classification_report(y_true, y_pred, target_names=target_names))

# LogisticRegression
from sklearn.linear_model import LogisticRegression

# modeli oluşturma ve eğitim
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train,y_train)
score.append(lr.score(X_test,y_test)*100)
algorithms.append("Logistic Regression")
print("Logistic Regression accuracy {}".format(lr.score(X_test,y_test)))

# karmaşıklık matrisi
from sklearn.metrics import confusion_matrix
y_pred=lr.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

# korelasyon grafiği
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Logistic Regression Confusion Matrix")
plt.show()
target_names=["Kötü","Yeterli","Orta","İyi","Çok iyi"]
print(classification_report(y_true, y_pred, target_names=target_names))

# Artificial Neural Networks

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# modeli oluşturma ve eğitim
sknet = MLPClassifier(hidden_layer_sizes=(15), learning_rate_init=0.05, max_iter=100)
sknet.fit(X_train, y_train)

score.append(sknet.score(X_test,y_test)*100)
algorithms.append("Artificial Neural Networks")

# karmaşıklık matrisi
y_pred = sknet.predict(X_test)
y_true=y_pred
cm=confusion_matrix(y_true,y_pred)

# korelasyon grafiği
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Artificial Neural Networks Confusion Matrix")
plt.show()
target_names=["Kötü","Yeterli","Orta","İyi","Çok iyi"]
print(classification_report(y_true, y_pred, target_names=target_names))

# sonuç
print (algorithms)
print (score)

x_pos = [i for i, _ in enumerate(algorithms)]

plt.bar(x_pos, score, color='#26A69A',edgecolor='black')
plt.xlabel("Algoritmalar")
plt.ylabel("Basari Yuzdeleri")
plt.title("Basari Siralamalar")

plt.xticks(x_pos, algorithms,rotation=90)
plt.show()











