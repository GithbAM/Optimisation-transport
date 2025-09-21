# Pipeline de Prédiction Régularité TGV

> Système de prédiction et optimisation de la régularité des trains TGV avec **MAE 14.25 pts** et dashboard Streamlit interactif

## ⚠️ Status : Proof of Concept

Dashboard fonctionnel avec quelques optimisations en cours sur la réactivité de l'interface. Pipeline ML validé et opérationnel.

## Démo Live

**Dashboard :** [Prédictions & Optimisation ML](https://optimisation-transport.streamlit.app/)

## Résultats

| Modèle | CV MAE | CV RMSE | **CV R²** | Hold-out MAE |
|--------|--------|---------|-----------|--------------|
| **Linear Regression** | **14.70 pts** | **18.37 pts** | **0.263** | **🎯 14.25 pts** |
| Random Forest | 15.41 pts | 19.26 pts | 0.204 | - |
| XGBoost | 15.25 pts | 19.10 pts | 0.220 | - |
| LightGBM | 15.07 pts | 18.80 pts | 0.241 | - |

**Signification Opérationnelle :**
- **Erreur moyenne 14.25 points** : Si la régularité réelle est 85%, le modèle prédit entre 70.75% et 99.25%

## Métriques Business & Techniques

- **Régularité Moyenne TGV** : 85.3%
- **Liaison la Plus Problématique** : Identifiée automatiquement
- **Impact CO₂** : Calcul basé sur 0.21 kg CO₂/km évité par report modal
- **Projections Financières** : Valorisation carbone à 85€/tonne CO₂
## Stack Technique & Compétences

**ML Ops & Data Science:**
- `scikit-learn` (LinearRegression, RandomForest, XGBoost, LightGBM)
- `pandas` / `numpy` (Traitement données SNCF temporelles)
- Feature Engineering Avancé (Lags, Moyennes Mobiles, Features Calendaires)
- Time Series Cross-Validation & Pipeline ML

**Déploiement & Production:**
- `streamlit` (Dashboard interactif multi-onglets)
- `plotly` (Visualisations dynamiques & projections)
- Déploiement Cloud (Streamlit Cloud)

## Fonctionnalités Clés

- **Prédiction Temps Réel** : Estimation de régularité par liaison et période
- **Simulateur d'Amélioration** : Impact d'investissements sur performance et CO₂
- **Projections Multi-Scénarios** : Horizons 1-10 ans avec croissance paramétrable  
- **Analyse de Priorités** : Identification variables critiques par liaison
- **Impact Environnemental** : Calcul automatique économies CO₂ (kg/an)
- **Interface Intuitive** : Dashboard professionnel avec métriques temps réel

## Architecture du Projet

```
transport/
├── data/
│   └── regularite-mensuelle-tgv-aqst.csv    # Dataset SNCF
├── models/
│   ├── best_punctuality_model.pkl           # Modèle final
│   ├── analyses.bz2                         # Analyses métier
│   └── linear_impact.bz2                    # Importance features
├── notebooks/
│   ├── 1_prédictions.ipynb                  # Pipeline principal
│   ├── old_01_pipeline.ipynb                # Pipeline développement
│   └── old_app.ipynb                        # Prototype dashboard
└── pages/
    └── 1_prédictions.py                     # Application Streamlit
```

## Utilisation

```python
# Pipeline automatisé complet
model = load_model()
df_hist = load_data()

# Prédiction pour une liaison
pred, ic = calculate_prediction(
    liaison="Paris → Lyon", 
    date_pred=datetime.date.today(),
    nb_trains=150,
    df_hist=df_hist,
    model=model
)

# Simulation d'amélioration
baseline = calculate_baseline_metrics(liaison, date_pred, nb_trains, df_hist, model)
simulation = simulate_improvement(baseline, improvement_pct=15, model, liaison, date_pred, df_hist)

print(f"Régularité prédite: {pred:.1f}%")
print(f"Impact amélioration: +{simulation['delta_reg']:.2f}%")
print(f"CO₂ évité/an: {simulation['co2_kg_an']:,.0f} kg")
```

## Installation & Lancement

```bash
# Installation dépendances
pip install streamlit pandas numpy plotly scikit-learn xgboost lightgbm joblib

# Lancement dashboard
streamlit run pages/1_prédictions.py
```
