import streamlit as st

def app():
    st.write("""
    # Modelo de predicción de ventas sell out
    En la siguiente web app se presentan los modelos a escala semanal, quincenal
    y mensual de las ventas sell out.

    El aplicativo realiza una consulta directa a la base de datos y los filtra
    con el menu desplegable, luego compila el modelo considerando las variables
    ventas en 2 años anteriores y las caracteristicas de segmento, region, categoria,
    marca, ya que estas variables muestran unas caracteristicas de comportamiento repetitivo
    a lo largo del tiempo.

    Seleccione cual dimension de tiempo desea predecir.
    """)
