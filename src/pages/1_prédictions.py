#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib, shap, datetime as dt
from sklearn.linear_model import LinearRegression


# In[ ]:


st.set_page_config(page_title="Prédictions & Optimisation ML", page_icon="🤖", layout="wide")

# CHARGEMENT 
@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parents[2] / "models" / "best_punctuality_model.pkl"
    return joblib.load(model_path)

@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parents[2] / "data" / "regularite-mensuelle-tgv-aqst.csv"
    df = pd.read_csv(data_path, sep=';', encoding='utf-8')
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["periode"] = pd.to_datetime(df["date"], format="%Y-%m")
    df['mois'] = df['periode'].dt.month
    df['jour_semaine'] = df['periode'].dt.dayofweek
    df["taux_regularite"] = 100 - df["nombre_de_trains_en_retard_au_départ"] / df["nombre_de_circulations_prévues"] * 100
    df['taux_retard'] = 100 - df['taux_regularite']
    df['liaisons'] = df["gare_de_départ"] + " → " + df["gare_d'arrivée"]
    # features lag
    df = df.sort_values('periode')
    df['lag_1'] = df['taux_regularite'].shift(1)
    df['lag_7'] = df['taux_regularite'].shift(7)
    df['ma_7']  = df['taux_regularite'].rolling(7).mean()
    drop_list = ['commentaire_annulations', 'commentaire_retards_au_départ', "commentaire_retards_à_l'arrivée"]
    df.drop(drop_list, axis=1, inplace=True)
    df = df.dropna()
    return df

def get_historical_data_for_prediction(liaison, date_pred, df_hist):
    """Récupère les données historiques pour une liaison et date données"""
    df_lia = df_hist[df_hist['liaisons'] == liaison]
    
    if df_lia.empty:
        return None
        
    wanted_month = pd.to_datetime(date_pred).month
    monthly_data = df_lia[df_lia['periode'].dt.month == wanted_month]
    
    if monthly_data.empty:
        # Si pas de données pour ce mois, prendre la dernière ligne disponible
        return df_lia.tail(1).iloc[0]
    
    return monthly_data.tail(1).iloc[0]

def calculate_prediction(liaison, date_pred, nb_trains, df_hist, model):
    """Calcule la prédiction de régularité"""
    row = get_historical_data_for_prediction(liaison, date_pred, df_hist)
    
    if row is None:
        return None, None
        
    wanted_month = pd.to_datetime(date_pred).month
    
    X = np.array([[wanted_month,
                   date_pred.weekday(),
                   nb_trains,
                   row['lag_1'],
                   row['lag_7'],
                   row['ma_7']]])
    
    pred = model.predict(X)[0]
    ic = 3.5  # Basé sur MAE typique des modèles de régularité
    
    return pred, ic

def calculate_baseline_metrics(liaison, date_pred, nb_trains, df_hist, model):
    """Calcule les métriques de base pour une liaison donnée, utilisée pour calculer l'impact réel d'une liaison"""
    pred_base, ic = calculate_prediction(liaison, date_pred, nb_trains, df_hist, model)
    
    if pred_base is None:
        return None
        
    # Statistiques historiques de cette liaison
    df_liaison = df_hist[df_hist['liaisons'] == liaison]
    
    baseline_data = {
        'pred_base': pred_base,
        'ic': ic,
        'historical_avg': df_liaison['taux_regularite'].mean() if not df_liaison.empty else pred_base,
        'historical_std': df_liaison['taux_regularite'].std() if not df_liaison.empty else 2.0,
        'nb_trains_base': nb_trains
    }
    
    return baseline_data

def simulate_improvement(baseline_data, improvement_pct, model, liaison, date_pred, df_hist):
    """Simule l'amélioration pour une liaison spécifique"""
    if baseline_data is None:
        return None
        
    # Nouveaux paramètres
    new_nb_trains = int(baseline_data['nb_trains_base'] * (1 + improvement_pct/100))
    
    # Prédiction avec plus de trains
    pred_more_trains, _ = calculate_prediction(liaison, date_pred, new_nb_trains, df_hist, model)
    
    if pred_more_trains is None:
        pred_more_trains = baseline_data['pred_base']
    
    # Simulation amélioration opérationnelle (plus de trains peut = plus de retards OU mieux si investissement)
    # On simule que l'investissement dans + de trains s'accompagne d'améliorations
    operational_improvement = improvement_pct * 0.15  # 1.5% régularité par 10% investissement
    
    pred_final = pred_more_trains + operational_improvement
    pred_final = min(pred_final, 100)  # Cap à 100%
    
    delta_reg = pred_final - baseline_data['pred_base']
    co2_impact = calculate_co2_impact(delta_reg, new_nb_trains)
    
    return {
        'new_trains': new_nb_trains,
        'pred_final': pred_final,
        'delta_reg': delta_reg,
        'co2_kg_an': co2_impact
    }

def calculate_co2_impact(delta_reg, new_trains):
    """Calcule l'impact CO2 d'une amélioration"""
    CO2_VOITURE_PAR_TRAJET_KM = 0.21
    DISTANCE_MOY = 220
    PLACES_TGV = 150
    TAUX_REPORT = 0.75
    FACTEUR_AN = 365
    
    trajets_sup = delta_reg / 100 * new_trains * PLACES_TGV * TAUX_REPORT * FACTEUR_AN
    km_evites = trajets_sup * DISTANCE_MOY
    co2_kg_an = km_evites * CO2_VOITURE_PAR_TRAJET_KM
    
    return co2_kg_an

model   = load_model()
raw_model = model.named_steps["reg"]

df_hist = load_data()
analyses_path = Path(__file__).resolve().parents[2] / "models" / "analyses.bz2"
analyses = joblib.load(analyses_path)

# CONSTANTS 
LABELS = {
    "mois"                        : "Mois",
    "jour_semaine"                : "Jour de la semaine",
    "nombre_de_circulations_prévues" : "Trains programmés",
    "lag_1"                       : "Régularité mois précédent",
    "lag_7"                       : "Régularité il y a 7 mois",
    "ma_7"                        : "Moyenne mobile 7 mois"
}


# Résumé métier 
st.info(
    f" **Régularité moyenne TGV** : `{df_hist.taux_regularite.mean():.1f} %`  \n"
    f" **Liaison la plus en retard** : "
    f"`{analyses['sncf_performance']['problematic_liaisons'].iloc[0]['liaisons']}` "
    f"(`{analyses['sncf_performance']['problematic_liaisons'].iloc[0]['taux_retard']:.1f} %`)"
)

# SIDEBAR 
with st.sidebar:
    st.markdown("### 🎛️ Paramètres")
    date_pred = st.date_input("Date de prédiction", value=dt.date.today())
    liaison   = st.selectbox("Liaison", sorted(df_hist['liaisons'].unique()))
    nb_trains = st.number_input("Nb trains programmés", min_value=1, max_value=500, value=150)

# TABS 
tab1, tab2, tab3, tab4 = st.tabs(["Prédiction", "Simulateur", "Projections", "Priorités"])

# Prédiction temps réel
with tab1:
    st.markdown("#### Prédiction de régularité")
    
    baseline = calculate_baseline_metrics(liaison, date_pred, nb_trains, df_hist, model)
    
    if baseline is None:
        st.error(f"Aucune donnée disponible pour **{liaison}**")
        st.info("Sélectionnez une autre liaison dans la barre latérale")
    else:
        pred = baseline['pred_base']
        ic = baseline['ic']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Régularité prédite", f"{pred:.1f}%")
        
        with col2:
            st.metric("Retard estimé", f"{100-pred:.1f}%", 
                     help="Complément de la régularité (100% - régularité)")
        
        with col3:
            st.metric("Moy. historique", f"{baseline['historical_avg']:.1f}%",
                     delta=f"{pred - baseline['historical_avg']:+.1f}%")
        
        with col4:
            st.metric("Intervalle 95%", f"±{ic:.1f}%",
                     help=f"[{pred-ic:.1f}% - {pred+ic:.1f}%]")

# Simulateur d’amélioration
with tab2:
    st.markdown("#### Simulateur d'améliorations")
    
    baseline = calculate_baseline_metrics(liaison, date_pred, nb_trains, df_hist, model)
    
    if baseline is None:
        st.error(f"Impossible de simuler pour **{liaison}**")
    else:
        # Slider SANS valeur par défaut fixe
        st.markdown(f"**Liaison analysée** : `{liaison}`")
        st.markdown(f"**Période** : `{date_pred.strftime('%Y-%m')}`")
        
        improvement = st.slider(
            "Investissement d'amélioration (%)", 
            min_value=0, max_value=50, value=0, step=5,  # VALEUR PAR DÉFAUT 0
            help="Augmentation budget → + trains + améliorations opérationnelles"
        )
        
        if improvement == 0:
            st.info("Ajustez le curseur pour voir l'impact des améliorations")
        else:
            simulation = simulate_improvement(baseline, improvement, model, liaison, date_pred, df_hist)
            
            if simulation:
                st.markdown("---")
                st.markdown("### Résultats de la simulation")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Impact opérationnel**")
                    st.metric("Trains/mois", f"{simulation['new_trains']}", 
                             delta=f"+{simulation['new_trains'] - baseline['nb_trains_base']}")
                    st.metric("Δ Régularité", f"{simulation['delta_reg']:+.2f}%",
                             delta=f"{simulation['delta_reg']:+.2f}%",
                             delta_color="normal" if simulation['delta_reg'] > 0 else "inverse")
                
                with col2:
                    st.markdown("**Impact environnemental**")
                    st.metric("CO₂ évité/an", f"{simulation['co2_kg_an']:,.0f} kg")
                    st.metric("Valeur carbone", f"{simulation['co2_kg_an']/1000*85:,.0f} €",
                             help="Basé sur 85€/tonne CO₂")

# Projections futures
with tab3:
    # Recalculer co2_kg_an pour cet onglet
    pred_base, _ = calculate_prediction(liaison, date_pred, nb_trains, df_hist, model)
    
    if pred_base is not None:
        # Simulation d'amélioration pour les projections (utiliser valeurs par défaut)
        default_freq = 15
        new_trains_proj = int(nb_trains * (1 + default_freq/100))
        pred_improved_proj, _ = calculate_prediction(liaison, date_pred, new_trains_proj, df_hist, model)
        pred_improved_proj += default_freq * 0.1  # boost opérationnel
        delta_reg_proj = pred_improved_proj - pred_base
        base_co2 = calculate_co2_impact(delta_reg_proj, new_trains_proj)
        
        years = st.slider("Horizon (ans)", 1, 10, 5)
        growth = st.selectbox("Scénario", ["Modéré +5 %", "Ambitieux +8 %", "Révolution +12 %"])
        rate = {"Modéré +5 %":0.05, "Ambitieux +8 %":0.08, "Révolution +12 %":0.12}[growth]
        
        future = [base_co2 * (1+rate)**y for y in range(1,years+1)]
        cumul = np.cumsum(future)
        proj_df = pd.DataFrame({"an":range(1,years+1), "co2":future, "cumul":cumul})
        
        fig = px.line(proj_df, x="an", y="cumul", markers=True, 
                      title=f"CO₂ évité cumulé ({growth}) - Liaison: {liaison}")
        st.plotly_chart(fig, use_container_width=True)
        
        recap = pd.DataFrame({
            "Année": [dt.date.today().year + y for y in range(1, years+1)],
            "CO₂ évité (t)": [f"{v/1000:.0f}" for v in future], 
            "CO₂ cumulé (t)": [f"{v/1000:.0f}" for v in cumul]
        })
        st.dataframe(recap, hide_index=True, use_container_width=True)
    else:
        st.warning("Impossible de calculer les projections sans données historiques")

# Top priorités
with tab4:
    st.markdown("#### Priorités d'action")
    
    baseline = calculate_baseline_metrics(liaison, date_pred, nb_trains, df_hist, model)
    
    if baseline is None:
        st.warning("Analyse impossible sans données")
    else:
        if isinstance(raw_model, LinearRegression):
            # Importance générale
            st.markdown("##### Importance des variables (globale)")
            try:
                imp = joblib.load("linear_impact.bz2")
                for i, (_, row) in enumerate(imp.head(3).iterrows()):
                    with st.expander(f"#{i+1} - {LABELS.get(row['feature'], row['feature'])}"):
                        if row["impact"] > 0:
                            st.success(f"**Augmenter** cette variable → gain **+{row['impact']:.1f} pts** par écart-type")
                        else:
                            st.error(f"**Diminuer** cette variable → gain **{abs(row['impact']):.1f} pts** par écart-type")
                            
                        # Contexte spécifique à la liaison
                        if row['feature'] == 'nombre_de_circulations_prévues':
                            st.info(f"**Pour {liaison}** : actuellement {nb_trains} trains programmés")
                        elif row['feature'] in ['lag_1', 'lag_7', 'ma_7']:
                            hist_val = baseline.get('historical_avg', 0)
                            st.info(f"**Pour {liaison}** : régularité historique moyenne {hist_val:.1f}%")
                            
            except FileNotFoundError:
                st.error("Fichier d'analyse d'importance non trouvé")
        else:
            st.info("Analyse d'importance disponible uniquement pour la régression linéaire")
            
        # Recommandations spécifiques
        st.markdown("#####Recommandations pour cette liaison")
        
        if baseline['pred_base'] < baseline['historical_avg']:
            st.warning(f"Performance prédite ({baseline['pred_base']:.1f}%) inférieure à la moyenne historique ({baseline['historical_avg']:.1f}%)")
            st.markdown("**Actions recommandées** :")
            st.markdown("- Maintenance préventive accrue")
            st.markdown("- Analyse des causes de dégradation")
        else:
            st.success(f"Performance prédite ({baseline['pred_base']:.1f}%) supérieure à la moyenne")
            st.markdown("**Actions d'optimisation** :")
            st.markdown("- Capitaliser sur les bonnes pratiques") 
            st.markdown("- Augmenter la fréquence si demande")   

