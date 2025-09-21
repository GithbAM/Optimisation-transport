#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib, shap, datetime as dt

st.set_page_config(page_title="Transport Public France", page_icon="🚆", layout="wide")

# SESSION STATE
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'results' not in st.session_state:
    st.session_state.results = None

# TITRE
st.markdown("# 🚆 Transport Public France – Prédictions ML")
st.caption("Upload des données SNCF & IDF puis accès aux prédictions et optimisation intelligente.")

# UPLOAD
with st.sidebar:
    st.markdown("### 📁 Upload datasets")
    st.file_uploader("SNCF – Régularité TGV (CSV)", type="csv", key="sncf")
    if st.button("🚀 Lancer analyse", type="primary"):
        st.session_state.data_loaded = True
        st.success("Analyses prêtes – rendez-vous dans Prédictions")

# MODE D’EMPLOI
if not st.session_state.data_loaded:
    st.info("👈 Upload du fichier via la barre latérale puis cliquez sur **Lancer analyse**.")
    with st.expander("Formats attendus"):
        st.markdown("""
        **SNCF** : colonnes `periode`, `service`, `liaisons`, `nombre_de_trains_programmes`, `taux_regularite`  
        """)
else:
    st.success("✅ Données chargées – rendez-vous dans la page **Prédictions**")

