# Application Streamlit unique : regroupe les 4 modèles
# (chaque modèle a ses onglets 'Simple Prédiction' et 'Prédiction multiple')
from pathlib import Path

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Prédictions Machine Learning", page_icon="🤖", layout="centered")

# Dossier des modèles
ML = Path(__file__).parent / "Machine Learning"

# Style : en-tête des pages et cartes
st.markdown("""
<style>
.block-container {padding-top: 2.5rem;}
.hero {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    color: white; border-radius: 16px; padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;
}
.hero .icon {font-size: 2.4rem; line-height: 1;}
.hero h1 {color: white; font-size: 1.7rem; margin: .5rem 0 .3rem 0; padding: 0;}
.hero p {color: rgba(255,255,255,.88); margin: 0; font-size: .97rem;}
.hero .tag {
    display: inline-block; background: rgba(255,255,255,.2); border-radius: 999px;
    padding: .15rem .7rem; font-size: .78rem; font-weight: 600; letter-spacing: .02em;
}
.result-label {color: #64748b; font-size: .85rem; margin-bottom: .2rem;}
.result-value {font-size: 2.1rem; font-weight: 700; color: #0f172a; line-height: 1.2;}
</style>
""", unsafe_allow_html=True)


# Charger les objets sauvegardés d'un dossier (une seule fois grâce au cache)
@st.cache_resource
def load_objects(folder, model_file):
    d = ML / folder
    encoders = joblib.load(d / 'encoders.joblib')
    model = joblib.load(d / model_file)
    uniques = joblib.load(d / 'uniques.joblib')
    scaler = joblib.load(d / 'scaler.joblib')
    return encoders, model, uniques, scaler


# Exemple de CSV (5 premières lignes du jeu de données, sans la cible)
@st.cache_data
def sample_csv(csv_path, n_features):
    return pd.read_csv(ML / csv_path).iloc[:5, :n_features].to_csv(index=False).encode('utf-8')


# En-tête d'une page
def header(icon, title, tag, description):
    st.markdown(f"""
    <div class="hero">
        <div class="icon">{icon}</div>
        <h1>{title}</h1>
        <p><span class="tag">{tag}</span></p>
        <p style="margin-top:.6rem">{description}</p>
    </div>
    """, unsafe_allow_html=True)


# Carte de résultat pour la régression
def show_value(label, value):
    with st.container(border=True):
        st.markdown(f'<div class="result-label">{label}</div>'
                    f'<div class="result-value">{value}</div>', unsafe_allow_html=True)


# Carte de résultat pour la classification (classe prédite + probabilités)
def show_class(label, prediction, probas):
    with st.container(border=True):
        col1, col2 = st.columns([2, 3], vertical_alignment='center')
        with col1:
            st.markdown(f'<div class="result-label">{label}</div>'
                        f'<div class="result-value">{prediction}</div>', unsafe_allow_html=True)
        with col2:
            for classe, p in sorted(probas.items(), key=lambda x: -x[1]):
                st.progress(float(p), text=f"{classe} — {p:.0%}")


# Fonction de prédiction multiple (commune aux 4 modèles)
def Pred_func_csv(df, pred_func, n_features, target_name):
    predictions = []
    # Boucle sur les lignes du dataframe
    for row in df.iloc[:, :n_features].values:
        predictions.append(pred_func(*row))
    df[target_name] = predictions
    return df


# Onglet de prédiction multiple (commun aux 4 modèles)
def multiple_tab(key, columns, csv_path, pred_func, target_name, classification, unit=''):
    n_features = len(columns)

    # Étape 1 : format attendu
    with st.container(border=True):
        st.markdown("**1. Préparer le fichier**")
        st.caption("Le fichier CSV doit contenir ces colonnes, dans cet ordre :")
        st.markdown(" ".join(f"`{c}`" for c in columns))
        st.download_button("Télécharger un exemple de CSV", data=sample_csv(csv_path, n_features),
                           file_name=f'exemple_{key}.csv', mime='text/csv', icon=":material/download:",
                           on_click='ignore', key=f'sample_{key}')

    # Étape 2 : chargement
    with st.container(border=True):
        st.markdown("**2. Charger le fichier**")
        file = st.file_uploader('Charger un fichier CSV', type=['csv'], key=f'file_{key}',
                                label_visibility='collapsed')

    if file is None:
        st.session_state.pop(f'pred_{key}', None)
        return

    # Lire le fichier csv
    df = pd.read_csv(file)
    if df.shape[1] < n_features:
        st.error(f"Le fichier contient {df.shape[1]} colonnes, il en faut au moins {n_features}.")
        return

    with st.expander(f"Aperçu des données chargées ({len(df)} lignes)"):
        st.dataframe(df.head(), hide_index=True)

    if st.button('Prédire', key=f'btn_{key}', type='primary', width='stretch', icon=":material/bolt:"):
        try:
            st.session_state[f'pred_{key}'] = Pred_func_csv(df, pred_func, n_features, target_name)
        except ValueError as e:
            st.error(f"Une valeur du fichier n'est pas reconnue par le modèle : {e}")

    # Étape 3 : résultats (gardés en mémoire entre deux actions)
    df_pred = st.session_state.get(f'pred_{key}')
    if df_pred is not None:
        st.markdown("**3. Résultats**")
        if classification:
            counts = df_pred[target_name].value_counts()
            cols = st.columns(len(counts) + 1)
            cols[0].metric("Lignes prédites", len(df_pred), border=True)
            for col, (classe, n) in zip(cols[1:], counts.items()):
                col.metric(classe, n, border=True)
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Lignes prédites", len(df_pred), border=True)
            col2.metric("Moyenne", f"{df_pred[target_name].mean():.2f} {unit}", border=True)
            col3.metric("Min – Max", f"{df_pred[target_name].min():.1f} – {df_pred[target_name].max():.1f} {unit}",
                        border=True)
        st.dataframe(df_pred, hide_index=True,
                     column_config={target_name: st.column_config.Column(f"✨ {target_name}")})
        # Télécharger le fichier des prédictions
        st.download_button(label='Télécharger le fichier CSV',
                           data=df_pred.to_csv(index=False).encode('utf-8'),
                           file_name='predictions.csv',
                           mime='text/csv', type='primary', width='stretch',
                           icon=":material/download:", on_click='ignore', key=f'dl_{key}')


# Onglets d'une page
def tabs():
    return st.tabs([':material/edit_note: Simple Prédiction', ':material/upload_file: Prédiction multiple'])


# =====================================================================
# 1. CLASSIFICATION : état d'un téléphone
# =====================================================================
def page_telephone():
    _, rf_model, uniques, scaler = load_objects('CLASSIFICATION', 'rf_model.joblib')

    # Vecteur normalisé
    # (encoders.joblib contient 3 fois le même LabelEncoder, ajusté sur 'etat' :
    #  on encode donc avec l'ordre trié de uniques, identique à celui du LabelEncoder)
    def to_x(prix, adresse, marque, dim_ecr, ram, stockage):
        # Encoder les valeurs des adresse et marque
        adresse = uniques[0].index(adresse)
        marque = uniques[1].index(marque)
        # vecteur des valeurs numériques
        x_new = np.array([prix, adresse, marque, dim_ecr, ram, stockage], dtype=float).reshape(1, -1)
        # Normaliser
        return scaler.transform(x_new)

    # Fonction de prédiction
    def Pred_func(*args):
        # Prédire puis décoder la classe prédite
        y_pred = rf_model.predict(to_x(*args))
        return uniques[2][int(y_pred[0])]

    header("📱", "État d'un téléphone", "Classification · Random Forest",
           """Prédit l'état d'un téléphone (D'occasion, Neuf, Réconditionné, Venant) à partir
           de son prix, de l'adresse, de la marque, de la taille de l'écran, de la RAM et du stockage.""")
    tab1, tab2 = tabs()

    with tab1:
        with st.form('form_telephone'):
            st.markdown("**Annonce**")
            col1, col2, col3 = st.columns(3)
            prix = col1.number_input('Prix (FCFA)', min_value=0, value=250000, step=5000)
            adresse = col2.selectbox('Adresse', options=uniques[0])
            marque = col3.selectbox('Marque', options=uniques[1])
            st.markdown("**Caractéristiques**")
            col1, col2, col3 = st.columns(3)
            dim_ecr = col1.number_input("Écran (pouces)", min_value=0.0, value=6.0, step=0.1)
            ram = col2.number_input('RAM (Go)', min_value=1, value=8, step=1)
            stockage = col3.number_input('Stockage (Go)', min_value=1, value=128, step=16)
            submit = st.form_submit_button('Prédire', type='primary', width='stretch', icon=":material/bolt:")
        if submit:
            x = to_x(prix, adresse, marque, dim_ecr, ram, stockage)
            probas = dict(zip(uniques[2], rf_model.predict_proba(x)[0]))
            show_class("État prédit", Pred_func(prix, adresse, marque, dim_ecr, ram, stockage), probas)

    with tab2:
        multiple_tab('telephone', ['prix', 'adresse', 'marque', 'dim_ecr', 'ram', 'stockage'],
                     'CLASSIFICATION/Telephone_data.csv', Pred_func, 'etat_predit', classification=True)


# =====================================================================
# 2. CLASSIFICATION / DEVOIR : voiture Venant ou D'occasion
# =====================================================================
def page_car_origin():
    _, dt_model, uniques, scaler = load_objects('CLASSIFICATION/DEVOIR', 'dt_model.joblib')

    # Vecteur normalisé
    # (encoders.joblib contient 4 fois le même LabelEncoder, ajusté sur 'Venant_Occasion' :
    #  on encode donc avec l'ordre trié de uniques, identique à celui du LabelEncoder)
    def to_x(Marque, Annee, Boite_Vitesse, Adresse, Prix):
        # Encoder les valeurs des Marque, Boite_Vitesse et Adresse
        Marque = uniques[0].index(Marque)
        Boite_Vitesse = uniques[1].index(Boite_Vitesse)
        Adresse = uniques[2].index(Adresse)
        # vecteur des valeurs numériques
        x_new = np.array([Marque, Annee, Boite_Vitesse, Adresse, Prix], dtype=float).reshape(1, -1)
        # Normaliser
        return scaler.transform(x_new)

    # Fonction de prédiction
    def Pred_func(*args):
        # Prédire puis décoder la classe prédite
        y_pred = dt_model.predict(to_x(*args))
        return uniques[3][int(y_pred[0])]

    header("🚙", "Provenance d'une voiture", "Classification · Decision Tree · Devoir",
           """Prédit si une voiture est importée (« Venant ») ou d'occasion (« D'occasion ») à partir
           de sa marque, de l'année, de la boîte de vitesse, de l'adresse et du prix.""")
    tab1, tab2 = tabs()

    with tab1:
        with st.form('form_car_origin'):
            st.markdown("**Véhicule**")
            col1, col2, col3 = st.columns(3)
            Marque = col1.selectbox('Marque', options=uniques[0])
            Annee = col2.number_input('Année', min_value=1990, max_value=2030, value=2015, step=1)
            Boite_Vitesse = col3.selectbox('Boîte de vitesse', options=uniques[1])
            st.markdown("**Annonce**")
            col1, col2 = st.columns(2)
            Adresse = col1.selectbox('Adresse', options=uniques[2])
            Prix = col2.number_input('Prix (FCFA)', min_value=0, value=7800000, step=100000)
            submit = st.form_submit_button('Prédire', type='primary', width='stretch', icon=":material/bolt:")
        if submit:
            x = to_x(Marque, Annee, Boite_Vitesse, Adresse, Prix)
            probas = dict(zip(uniques[3], dt_model.predict_proba(x)[0]))
            show_class("Provenance prédite", Pred_func(Marque, Annee, Boite_Vitesse, Adresse, Prix), probas)

    with tab2:
        multiple_tab('car_origin', ['Marque', 'Année', 'Boite_Vitesse', 'Adresse', 'Prix'],
                     'CLASSIFICATION/DEVOIR/Car_Data (1).csv', Pred_func, 'Venant_Occasion_predit',
                     classification=True)


# =====================================================================
# 3. REGRESSION : prix d'un trajet de taxi
# =====================================================================
def page_taxi():
    encoders, rf_model, _, scaler = load_objects('REGRESSION', 'rf_model.joblib')
    # uniques.joblib a été sauvegardé après l'encodage : on prend les classes des encodeurs
    uniques = [enc.classes_.tolist() for enc in encoders]

    # Fonction de prédiction
    def Pred_func(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                  Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes):
        # Encoder les valeurs des Time_of_Day, Day_of_Week, Traffic_Conditions et Weather
        Time_of_Day = encoders[0].transform([Time_of_Day])[0]
        Day_of_Week = encoders[1].transform([Day_of_Week])[0]
        Traffic_Conditions = encoders[2].transform([Traffic_Conditions])[0]
        Weather = encoders[3].transform([Weather])[0]
        # vecteur des valeurs numériques
        x_new = np.array([Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                          Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes],
                         dtype=float).reshape(1, -1)
        # Normaliser
        x_new = scaler.transform(x_new)
        # Prédire et arrondir
        y_pred = rf_model.predict(x_new)
        return round(float(y_pred[0]), 2)

    header("🚕", "Prix d'un trajet en taxi", "Régression · Random Forest",
           """Prédit le prix d'un trajet à partir de la distance, du moment, du jour, du nombre de passagers,
           du trafic, de la météo, des tarifs et de la durée du trajet.""")
    tab1, tab2 = tabs()

    with tab1:
        with st.form('form_taxi'):
            st.markdown("**Trajet**")
            col1, col2, col3 = st.columns(3)
            Trip_Distance_km = col1.number_input('Distance (km)', min_value=0.0, value=25.0, step=0.5)
            Trip_Duration_Minutes = col2.number_input('Durée (minutes)', min_value=0.0, value=60.0, step=1.0)
            Passenger_Count = col3.number_input('Passagers', min_value=1, max_value=4, value=2, step=1)
            st.markdown("**Conditions**")
            col1, col2, col3, col4 = st.columns(4)
            Time_of_Day = col1.selectbox('Moment', options=uniques[0])
            Day_of_Week = col2.selectbox('Jour', options=uniques[1])
            Traffic_Conditions = col3.selectbox('Trafic', options=uniques[2])
            Weather = col4.selectbox('Météo', options=uniques[3])
            st.markdown("**Tarifs**")
            col1, col2, col3 = st.columns(3)
            Base_Fare = col1.number_input('Tarif de base ($)', min_value=0.0, value=3.5, step=0.1)
            Per_Km_Rate = col2.number_input('Tarif par km ($)', min_value=0.0, value=1.2, step=0.05)
            Per_Minute_Rate = col3.number_input('Tarif par minute ($)', min_value=0.0, value=0.3, step=0.01)
            submit = st.form_submit_button('Prédire', type='primary', width='stretch', icon=":material/bolt:")
        if submit:
            y_pred = Pred_func(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                               Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes)
            show_value("Prix estimé du trajet", f"{y_pred:.2f} $")

    with tab2:
        multiple_tab('taxi', ['Trip_Distance_km', 'Time_of_Day', 'Day_of_Week', 'Passenger_Count',
                              'Traffic_Conditions', 'Weather', 'Base_Fare', 'Per_Km_Rate',
                              'Per_Minute_Rate', 'Trip_Duration_Minutes'],
                     'REGRESSION/sample_df.csv', Pred_func, 'Trip_Price', classification=False, unit='$')


# =====================================================================
# 4. REGRESSION / DEVOIR : prix de vente d'une voiture
# =====================================================================
def page_car_price():
    encoders, rf_model, uniques, scaler = load_objects('REGRESSION/DEVOIR', 'rf_model.joblib')

    # Fonction de prédiction
    def Pred_func(Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age):
        # Encoder les valeurs des Fuel_Type, Seller_Type et Transmission
        Fuel_Type = encoders[0].transform([Fuel_Type])[0]
        Seller_Type = encoders[1].transform([Seller_Type])[0]
        Transmission = encoders[2].transform([Transmission])[0]
        # vecteur des valeurs numériques
        x_new = np.array([Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age],
                         dtype=float).reshape(1, -1)
        # Normaliser
        x_new = scaler.transform(x_new)
        # Prédire et arrondir
        y_pred = rf_model.predict(x_new)
        return round(float(y_pred[0]), 2)

    header("🚗", "Prix de vente d'une voiture", "Régression · Random Forest · Devoir",
           """Prédit le prix de vente d'une voiture à partir du kilométrage, du prix actuel,
           du type de carburant, du type de vendeur, de la transmission et de l'âge.""")
    tab1, tab2 = tabs()

    with tab1:
        with st.form('form_car_price'):
            st.markdown("**Véhicule**")
            col1, col2, col3 = st.columns(3)
            Kms_Driven = col1.number_input('Kilométrage', min_value=0, value=27000, step=1000)
            Age = col2.number_input('Âge (années)', min_value=0, value=6, step=1)
            Present_Price = col3.number_input('Prix actuel (millions FCFA)', min_value=0.0, value=5.59, step=0.1)
            st.markdown("**Caractéristiques**")
            col1, col2, col3 = st.columns(3)
            Fuel_Type = col1.selectbox('Carburant', options=uniques[0])
            Transmission = col2.selectbox('Transmission', options=uniques[2])
            Seller_Type = col3.selectbox('Vendeur', options=uniques[1])
            submit = st.form_submit_button('Prédire', type='primary', width='stretch', icon=":material/bolt:")
        if submit:
            y_pred = Pred_func(Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age)
            show_value("Prix de vente estimé", f"{y_pred:.2f} millions FCFA")

    with tab2:
        multiple_tab('car_price', ['Kms_Driven', 'Present_Price', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Age'],
                     'REGRESSION/DEVOIR/Car_dataset.csv', Pred_func, 'Selling_Price', classification=False,
                     unit='M FCFA')


# =====================================================================
# Menu de navigation
# =====================================================================
PAGES = {
    "📱 État d'un téléphone": (page_telephone, "Classification · 90 % de bonnes prédictions"),
    "🚙 Provenance d'une voiture": (page_car_origin, "Classification · Devoir · 81 %"),
    "🚕 Prix d'un trajet en taxi": (page_taxi, "Régression · erreur moyenne 7,6 $"),
    "🚗 Prix de vente d'une voiture": (page_car_price, "Régression · Devoir · erreur moyenne 1,7 M FCFA"),
}

with st.sidebar:
    st.markdown("## 🤖 Prédictions ML")
    st.caption("4 modèles de Machine Learning déployés avec Streamlit.")
    st.divider()
    choix = st.radio("Choisir un modèle", list(PAGES.keys()),
                     captions=[caption for _, caption in PAGES.values()])
    st.divider()
    st.caption("Chaque modèle propose une prédiction simple (formulaire) "
               "et une prédiction multiple (fichier CSV).")

PAGES[choix][0]()
