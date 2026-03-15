import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("Segmentación de Clientes con K-Means")

st.write("Sube un archivo CSV para analizar los clientes")

# Subir archivo
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo is not None:

    df = pd.read_csv(archivo)

    st.subheader("Datos cargados")
    st.write(df)

    columnas = df.columns.tolist()

    col1 = st.selectbox("Selecciona la columna X", columnas)
    col2 = st.selectbox("Selecciona la columna Y", columnas)

    k = st.slider("Número de clusters", 2, 6, 3)

    X = df[[col1, col2]]

    modelo = KMeans(n_clusters=k)
    modelo.fit(X)

    df["Cluster"] = modelo.labels_

    st.subheader("Datos con clusters")
    st.write(df)

    fig, ax = plt.subplots()

    ax.scatter(df[col1], df[col2], c=df["Cluster"])
    ax.scatter(modelo.cluster_centers_[:,0], modelo.cluster_centers_[:,1], marker="X", s=200)

    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title("Segmentación de Clientes")

    st.pyplot(fig)
