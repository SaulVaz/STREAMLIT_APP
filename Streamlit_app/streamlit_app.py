import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

@st.cache_data
def cargar_datos():
    data = load_iris(as_frame=True)
    df = data.frame
    target_names = data.target_names
    return df, target_names

df, target_names = cargar_datos()

st.title("🌼 Clasificador de Iris con Streamlit")

if st.checkbox("Mostrar tabla de datos"):
    st.dataframe(df)

X = df.drop(columns='target')
y = df['target']

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

st.subheader("📊 Evaluación del Modelo")
y_pred = modelo.predict(X)
acc = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

st.write(f"Precisión del modelo: **{acc:.2f}**")

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Greens", xticklabels=target_names, yticklabels=target_names)
st.pyplot(fig)

st.subheader("🔍 Predice una flor")

sl = st.slider("Sepal Length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()))
sw = st.slider("Sepal Width (cm)", float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()))
pl = st.slider("Petal Length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()))
pw = st.slider("Petal Width (cm)", float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()))

if st.button("📌 Predecir especie"):
    input_data = [[sl, sw, pl, pw]]
    prediction = modelo.predict(input_data)[0]
    st.session_state['prediction'] = prediction 

if 'prediction' in st.session_state:
    pred = st.session_state['prediction']
    st.success(f"La flor predicha es: **{target_names[pred]}**")

st.subheader("ℹ️ Información Adicional")
st.write("Este modelo utiliza el conjunto de datos Iris para clasificar flores en tres especies diferentes: Setosa, Versicolor y Virginica.")
st.write("El modelo Random Forest se entrena con 100 árboles de decisión y se evalúa utilizando la precisión y la matriz de confusión.")
st.write("Puedes ajustar los parámetros del modelo y ver cómo afecta a la precisión.")
st.write("Los gráficos muestran la distribución de las características y la relación entre ellas.")
st.write("¡Explora y diviértete!")

st.subheader("📊 Gráfico de Dispersión")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='target', palette='Set2')
st.pyplot(fig)
st.subheader("📊 Gráfico de Boxplot")
fig, ax = plt.subplots()
sns.boxplot(data=df, x='target', y='sepal length (cm)', palette='Set2')
st.pyplot(fig)
st.subheader("📊 Gráfico de Violin")
fig, ax = plt.subplots()
sns.violinplot(data=df, x='target', y='sepal length (cm)', palette='Set2')
st.pyplot(fig)
st.subheader("📊 Gráfico de Distribución")
fig, ax = plt.subplots()
sns.histplot(data=df, x='sepal length (cm)', hue='target', kde=True, palette='Set2')
st.pyplot(fig)

#Prueba