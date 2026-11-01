# Deploy My Second Model With Streamlit

Application Streamlit qui regroupe **4 modèles de Machine Learning** : 2 de classification et 2 de régression.
Chaque modèle a deux onglets :

- **Simple Prédiction** : on remplit un formulaire et on obtient une prédiction.
- **Prédiction multiple** : on envoie un fichier CSV, on obtient une prédiction par ligne et on peut télécharger le résultat.

On choisit le modèle dans la barre latérale.

## Les modèles

| Modèle | Type | Données | Algorithme | Cible prédite |
|---|---|---|---|---|
| 📱 État d'un téléphone | Classification | `CLASSIFICATION/Telephone_data.csv` | Random Forest | `etat` : D'occasion, Neuf, Réconditionné, Venant |
| 🚙 Provenance d'une voiture (devoir) | Classification | `CLASSIFICATION/DEVOIR/Car_Data (1).csv` | Decision Tree | `Venant_Occasion` : Venant, D'occasion |
| 🚕 Prix d'un trajet de taxi | Régression | `REGRESSION/taxi_trip_pricing.csv` | Random Forest | `Trip_Price` ($) |
| 🚗 Prix de vente d'une voiture (devoir) | Régression | `REGRESSION/DEVOIR/Car_dataset.csv` | Random Forest | `Selling_Price` |

Les modèles ont été entraînés dans les notebooks Google Colab de chaque dossier.

## Structure du projet

```
.
├── app.py                      # application Streamlit (les 4 modèles)
├── .streamlit/config.toml      # thème de l'application (couleurs, police)
├── requirements.txt
└── Machine Learning/
    ├── CLASSIFICATION/
    │   ├── Clas_Log_Reg_Dec_Tree_Random_Forest.ipynb
    │   ├── Telephone_data.csv
    │   ├── rf_model.joblib, scaler.joblib, encoders.joblib, uniques.joblib
    │   └── DEVOIR/
    │       ├── Clas car data(1).ipynb
    │       ├── Car_Data (1).csv
    │       └── dt_model.joblib, scaler.joblib, encoders.joblib, uniques.joblib
    └── REGRESSION/
        ├── Reg_Mul_Reg_Ridge_Dec_Tree_Random_Forest.ipynb
        ├── taxi_trip_pricing.csv, sample_df.csv
        ├── rf_model.joblib, scaler.joblib, encoders.joblib, uniques.joblib
        └── DEVOIR/
            ├── rg car dataseet.ipynb
            ├── Car_dataset.csv
            └── rf_model.joblib, scaler.joblib, encoders.joblib, uniques.joblib
```

Dans chaque dossier, les fichiers `.joblib` sont :

- `*_model.joblib` : le meilleur modèle entraîné ;
- `scaler.joblib` : le scaler (MinMaxScaler en classification, RobustScaler en régression), appliqué avant chaque prédiction ;
- `encoders.joblib` : les `LabelEncoder` des variables catégorielles ;
- `uniques.joblib` : les valeurs possibles des variables catégorielles (pour les listes déroulantes).

## Installation et lancement

```bash
# créer un environnement virtuel et installer les dépendances
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt

# lancer l'application
streamlit run app.py
```

Avec `uv` : `uv venv && uv pip install -r requirements.txt && uv run streamlit run app.py`

L'application s'ouvre sur <http://localhost:8501>.

## Format des CSV pour la prédiction multiple

Les colonnes doivent être **dans l'ordre ci-dessous**, avec les mêmes noms que dans les données d'origine.
Une colonne cible placée en dernier est ignorée. Les CSV d'origine peuvent donc être envoyés tels quels pour tester.

| Modèle | Colonnes attendues (dans l'ordre) |
|---|---|
| 📱 Téléphone | `prix, adresse, marque, dim_ecr, ram, stockage` |
| 🚙 Provenance voiture | `Marque, Année, Boite_Vitesse, Adresse, Prix` |
| 🚕 Taxi | `Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions, Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes` |
| 🚗 Prix voiture | `Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age` |

Les valeurs des colonnes catégorielles doivent faire partie de celles vues à l'entraînement (par exemple `Morning`, `Afternoon`, `Evening` ou `Night` pour `Time_of_Day`).
Exemple prêt à l'emploi pour le taxi : `Machine Learning/REGRESSION/sample_df.csv`.

## Performances (sur les CSV complets)

| Modèle | Résultat |
|---|---|
| 📱 Téléphone | 90,3 % de bonnes prédictions |
| 🚙 Provenance voiture | 81 % de bonnes prédictions |
| 🚕 Taxi | erreur moyenne de 7,63 $ (prix moyen : 57,66 $) |
| 🚗 Prix voiture | erreur moyenne de 1,74 million FCFA (prix moyen : 10,2 millions FCFA) |

Ces chiffres sont calculés sur tout le jeu de données, donc aussi sur les lignes vues pendant l'entraînement.
Les scores sur les données de test se trouvent dans les notebooks.

## Remarques sur les fichiers joblib

Deux fichiers sauvegardés depuis les notebooks posent problème. `app.py` les contourne :

- **Classification (téléphone et voiture)** : `encoders.joblib` est inutilisable. Le notebook réutilise le même `LabelEncoder` dans la boucle, donc tous les encodeurs sauvegardés ne connaissent que la cible.
  `app.py` encode donc avec la position de la valeur dans `uniques.joblib`. C'est le même ordre alphabétique que `LabelEncoder`, donc le résultat est identique.
  Pour corriger le notebook, il faut créer l'encodeur **dans** la boucle :
  ```python
  encoders = []
  for col in cat_data.columns:
      encoder = LabelEncoder()          # un nouvel encodeur par colonne
      df[col] = encoder.fit_transform(df[col])
      encoders.append(encoder)
  ```
- **Régression taxi** : `uniques.joblib` a été sauvegardé après l'encodage. Il contient donc `0, 1, 2…` au lieu des libellés.
  `app.py` utilise à la place `encoder.classes_` de chaque encodeur.
