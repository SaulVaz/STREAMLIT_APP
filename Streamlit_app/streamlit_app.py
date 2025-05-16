import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

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
cax = ax.matshow(cm, cmap="Greens")
plt.title('Matriz de Confusión', pad=20)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.xticks(np.arange(3), target_names)
plt.yticks(np.arange(3), target_names)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], va='center', ha='center')
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

# Gráfico de Dispersión
st.subheader("📊 Gráfico de Dispersión")
fig, ax = plt.subplots()
colors = ['red', 'green', 'blue']
for i in range(3):
    especie = df[df['target'] == i]
    ax.scatter(especie['sepal length (cm)'], especie['sepal width (cm)'],
               label=target_names[i], alpha=0.7, c=colors[i])
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.legend()
st.pyplot(fig)

# Gráfico de Boxplot
st.subheader("📊 Gráfico de Boxplot")
fig, ax = plt.subplots()
data_to_plot = [df[df['target'] == i]['sepal length (cm)'] for i in range(3)]
ax.boxplot(data_to_plot, labels=target_names)
ax.set_title("Boxplot de Sepal Length")
st.pyplot(fig)

# Gráfico de Violin (reemplazado por histograma con densidad)
st.subheader("📊 Gráfico de Violin (simulado con líneas de densidad)")
fig, ax = plt.subplots()
for i in range(3):
    especie = df[df['target'] == i]['sepal length (cm)']
    especie.plot(kind='kde', ax=ax, label=target_names[i])
ax.set_title("Distribución KDE de Sepal Length")
ax.legend()
st.pyplot(fig)

# Histograma
st.subheader("📊 Histograma de Sepal Length")
fig, ax = plt.subplots()
for i in range(3):
    especie = df[df['target'] == i]['sepal length (cm)']
    ax.hist(especie, alpha=0.5, label=target_names[i])
ax.set_title("Histograma por Especie")
ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Frecuencia")
ax.legend()
st.pyplot(fig)
