import streamlit as st
from multiapp import MultiApp
from apps import vsigv_semanal, vsigv_quincena, vsigv_mes, home

# import your app modules here
app = MultiApp()

st.markdown("""
# Web app prediccion de ventas sell out protisa
Seleccione la escala temporal de prediccion (semanal, quincenal, mensual)
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Semanal", vsigv_semanal.app)
app.add_app("Quincena", vsigv_quincena.app)
app.add_app("Mensual", vsigv_mes.app)
# The main app
app.run()
