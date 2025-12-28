from flask import Flask, render_template, request, redirect, url_for
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import joblib
from flask import send_file, send_from_directory
from feature_selection import selection_features
from modeling import entrainer_et_comparer_cv
from optimization import optimiser_modele
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename
from exploration import analyser_donnees
from preprocessing import feature_engineering
from fraude.preprocessing import clean_data          
from fraude.smote_model import train_smote           
from fraude.threshold_tuning import threshold_tuning 
from fraude.shap_explain import explain             
from fraude.exploration import analyse_classes       


app = Flask(__name__)

UPLOAD_FOLDER = 'datauser'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def safe_float(value, default=0.0):
    try:
        return float(value) if value and str(value).strip() != "" else default
    except (ValueError, TypeError):
        return default
    
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join('static', 'plots'), exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/immo')
def immo():
    return render_template('immo.html')

@app.route("/fraude")
def fraude():
    return render_template("fraude.html")

@app.route('/fraude/upload', methods=['POST'])
def fraude_upload():

    # Vérification fichier
    if 'file' not in request.files:
        return "Aucun fichier trouvé"

    file = request.files['file']
    filename = secure_filename(file.filename)

    UPLOAD_FOLDER = os.path.join('uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Lecture & nettoyage
    df = pd.read_csv(filepath)
    df_clean = clean_data(df)

    nb_transactions = len(df_clean)

    # Analyse exploratoire (facultatif mais OK)
    ratio, pie_path, hist_path = analyse_classes(df_clean)

    # Préparer features (PAS DE TARGET ICI)
    X = df_clean.drop(columns=['is_fraud'], errors='ignore')

    # Respect des exclusions
    EXCLUDE_COLS = [
        "score_risque_marchand",
        "nb_tentatives_echouees",
        "montant_total_24h",
        "nb_trans_24h"
    ]
    X = X.drop(columns=[c for c in EXCLUDE_COLS if c in X.columns])

    # Charger modèle + seuil (OFFLINE)
    model = joblib.load("models/smote.pkl")
    with open("models/best_threshold.json") as f:
        best_thresh = json.load(f)["threshold"]

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= best_thresh).astype(int)

    df_results = X.copy()
    df_results["fraud_probability"] = y_prob
    df_results["is_fraud_pred"] = y_pred

    fraud_cases = df_results[df_results["is_fraud_pred"] == 1]

    top_features = []
    if not fraud_cases.empty:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(fraud_cases.drop(
            columns=["fraud_probability", "is_fraud_pred"]
        ))

        shap.summary_plot(
            shap_values[1],
            fraud_cases.drop(columns=["fraud_probability", "is_fraud_pred"]),
            show=False
        )

        os.makedirs("static/plots", exist_ok=True)
        plt.savefig("static/plots/shap_summary.png")
        plt.close()

        top_features = fraud_cases.columns[:5].tolist()

    DOWNLOAD_FOLDER = os.path.join('static', 'downloads')
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

    result_file = f"results_{filename}"
    df_results.to_csv(os.path.join(DOWNLOAD_FOLDER, result_file), index=False)

    return render_template(
        "fraude.html",
        nb_transactions=nb_transactions,
        nb_fraudes_detectees=int(df_results["is_fraud_pred"].sum()),
        table_results=df_results.head(10).to_html(classes='table table-striped'),
        pie_path=url_for('static', filename='plots/class_distribution.png'),
        hist_path=url_for('static', filename='plots/class_histogram.png'),
        shap_path=url_for('static', filename='plots/shap_summary.png'),
        best_thresh=best_thresh,
        top_features=top_features,
        result_file=result_file
    )
        
@app.route("/fraude/predict_ui")
def fraude_predict_ui():
    return render_template("fraude_predict.html")

@app.route("/fraude/get_prediction", methods=["POST"])
def fraude_get_prediction():
    # Exemple simplifié : récupérer les inputs
    montant = float(request.form.get("montant"))
    type_tx = request.form.get("type_tx")
    origine = request.form.get("origine")
    destination = request.form.get("destination")
    heure = request.form.get("heure")

    # Ici tu charges ton modèle de fraude bancaire
    model_fraude = joblib.load("models/modele_fraude_final.joblib")

    # Construction d'un DataFrame pour la prédiction
    X_input = pd.DataFrame([{
        "montant": montant,
        "type_tx": type_tx,
        "origine": origine,
        "destination": destination,
        "heure": heure
    }])

    # TODO : encoder les catégorielles / appliquer scaler si nécessaire
    prediction = model_fraude.predict(X_input)[0]

    return render_template("fraude_predict.html", prediction="Fraude" if prediction==1 else "Non frauduleuse")      

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Aucun fichier trouvé"
    
    file = request.files['file']
    if file.filename == '':
        return "Aucun fichier sélectionné"
    
    if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
             
            # 2. Chargement et Feature Engineering
            df_raw = pd.read_csv(filepath)
            df_clean, scaler  = feature_engineering(df_raw)
            
             # SAUVEGARDE DU FICHIER NETTOYÉ
            cleaned_filename = "data_paris_clean.csv"
            cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)
            df_clean.to_csv(cleaned_filepath, index=False) # Sauvegarde sans les index
            
            joblib.dump(scaler, 'models/scaler.joblib')
            # PHASE 3 : Sélection des Top 15 Features
            top_features, df_importance, rf_model, X_train, X_test, y_train, y_test = selection_features(df_clean)
            joblib.dump(top_features, 'models/top_features.joblib')

            

            # On lance la Phase 4 avec Cross-Validation
            df_comparatif = entrainer_et_comparer_cv(df_clean[top_features], df_clean['prix'])

            # On l'envoie au template
            models_html = df_comparatif.to_html(classes='table table-striped table-hover', index=False)
            
            result_opti = optimiser_modele(X_train[top_features], y_train, X_test[top_features], y_test)
            joblib.dump(result_opti['modele_final'], 'models/modele_paris_final.joblib')
    
            stats, hist, box, corr_img, corr_txt, miss, outliers, anomalies, nb, = analyser_donnees(df_clean)
        
            # Transmission au template dashboard.html
            return render_template('dashboard.html', 
                                stats=stats, 
                                hist_url=hist, 
                                box_url=box, 
                                corr_url=corr_img,
                                corr_table=corr_txt,
                                report_missing=miss,    
                                outliers=outliers,      
                                anomalies=anomalies,   
                                nb=nb,
                                top_features=top_features,
                                models_table=models_html,
                                opti=result_opti,
                                cleaned_file=cleaned_filename)

# Chargement des outils ML
model = joblib.load('models/modele_paris_final.joblib')
scaler = joblib.load('models/scaler.joblib')
top_features = joblib.load('models/top_features.joblib')

@app.route('/predict_ui')
def predict_ui():
    #return render_template('predict.html')
    return render_template('predict.html', inputs={})

@app.route('/get_prediction', methods=['POST'])
def get_prediction():
    # 1. Chargement des outils sauvegardés
    model = joblib.load('models/modele_paris_final.joblib')
    scaler = joblib.load('models/scaler.joblib')
    top_features = joblib.load('models/top_features.joblib')

    # 2. Récupération des données du formulaire
    surface = safe_float(request.form.get('surface'), 50.0)
    pieces = safe_float(request.form.get('pieces'), 2.0)
    
    # Création d'un dictionnaire temporaire pour mapper les saisies
    raw_inputs = {
        'surface_habitable': surface,
        'nb_pieces': pieces,
        'nb_chambres': safe_float(request.form.get('chambres'), 1.0),
        'etage': safe_float(request.form.get('etage'), 2.0),
        'distance_centre': safe_float(request.form.get('dist_centre'), 2.5),
        'distance_transport': safe_float(request.form.get('dist_trans'), 300.0),
        'annee_construction': safe_float(request.form.get('annee'), 1950.0),
        'score_commerces': safe_float(request.form.get('commerces'), 7.0),
        'ascenseur': safe_float(request.form.get('ascenseur'), 1.0),
        'parking': safe_float(request.form.get('parking'), 0.0),
        # Variable calculée
        'surface_par_piece': surface / pieces if pieces > 0 else surface
    }

    # 3. Préparation du DataFrame final (Initialisé à 0)
    # On s'assure d'avoir TOUTES les colonnes du Top 15, même les binaires
    X_input = pd.DataFrame(0, index=[0], columns=top_features)

    # Remplissage des numériques
    for col in raw_inputs:
        if col in X_input.columns:
            X_input[col] = raw_inputs[col]

    # 4. Gestion dynamique du One-Hot Encoding (Quartier, État, Type, Énergie)
    # On construit le nom de la colonne comme lors de l'entraînement (ex: 'quartier_Marais')
    categories = {
        'quartier': request.form.get('quartier'),
        'etat': request.form.get('etat'),
        'type_bien': request.form.get('type'),
        'classe_energie': request.form.get('energie')
    }

    for prefix, val in categories.items():
        col_name = f"{prefix}_{val}"
        if col_name in X_input.columns:
            X_input[col_name] = 1

    # 5. Normalisation avec le Scaler
    # num_cols = ['surface_habitable', 'nb_pieces', 'nb_chambres', 'etage', 
    #             'distance_centre', 'distance_transport', 'score_commerces', 'surface_par_piece']
    
    # cols_to_scale = [c for c in num_cols if c in X_input.columns]
    # if cols_to_scale:
    #     X_input[cols_to_scale] = scaler.transform(X_input[cols_to_scale])

    # 5. Normalisation avec le Scaler (Version Robuste)
    
    # On récupère la liste EXACTE des colonnes que le scaler attend (celles de l'entraînement)
    # feature_names_in_ est l'attribut de scikit-learn qui stocke ces noms
    expected_scaler_cols = scaler.feature_names_in_

    # On prépare un petit DataFrame temporaire pour le scaling
    # Il contient uniquement les colonnes numériques attendues par le scaler
    X_for_scaling = pd.DataFrame(0, index=[0], columns=expected_scaler_cols)

    # On remplit ce DataFrame avec les valeurs que nous avons (raw_inputs)
    for col in expected_scaler_cols:
        if col in raw_inputs:
            X_for_scaling[col] = raw_inputs[col]
        else:
            # Si une colonne manque (ex: n'est pas dans le Top 15), on met la médiane 
            # ou 0 pour ne pas fausser le calcul, mais le scaler doit la voir.
            X_for_scaling[col] = 0 

    # On transforme (scaling)
    X_scaled_values = scaler.transform(X_for_scaling)
    
    # On convertit le résultat en dictionnaire pour le réinjecter dans X_input
    scaled_dict = dict(zip(expected_scaler_cols, X_scaled_values[0]))

    # On met à jour X_input (le Top 15) avec les valeurs normalisées
    for col in X_input.columns:
        if col in scaled_dict:
            X_input[col] = scaled_dict[col]
    
    # 6. Prédiction
    prediction = model.predict(X_input)[0]

    return render_template('predict.html', 
                           prediction=round(prediction, 2), 
                           inputs=request.form)
    
# @app.route('/get_prediction', methods=['POST'])
# def get_prediction():
    # 1. Récupération des données brutes du formulaire
    surface = float(request.form.get('surface'))
    pieces = int(request.form.get('pieces'))
    
    data = {
        'surface_habitable': surface,
        'nb_pieces': pieces,
        'surface_par_piece': surface / pieces,
        'distance_centre': float(request.form.get('dist_centre')),
        'etage': int(request.form.get('etage')),
        'annee_construction': int(request.form.get('annee')),
        'nb_chambres': float(request.form.get('chambres')),
        'distance_transport': float(request.form.get('dist_trans')),
        'score_commerces': float(request.form.get('commerces')),
        # Variables catégorielles (Dummies)
        'type_bien_Loft': 1 if request.form.get('type') == 'Loft' else 0,
        'type_bien_Maison': 1 if request.form.get('type') == 'Maison' else 0,
        'etat_Neuf': 1 if request.form.get('etat') == 'Neuf' else 0,
        'etat_Correct': 1 if request.form.get('etat') == 'Correct' else 0,
        'etat_À rénover': 1 if request.form.get('etat') == 'Renover' else 0,
        'quartier_Marais': 1 if request.form.get('quartier') == 'Marais' else 0,
        'classe_energie_E': 1 if request.form.get('energie') == 'E' else 0
    }

    # 2. Création du DataFrame et remise dans l'ordre exact des colonnes
    X_input = pd.DataFrame([data])
    X_input = X_input[top_features] # On ne garde que les 15 variables choisies

    # 3. Application du Scaler (Uniquement sur les colonnes numériques)
    num_cols = ['surface_habitable', 'nb_pieces', 'nb_chambres', 'etage', 
                'distance_centre', 'distance_transport', 'score_commerces', 'surface_par_piece']
    X_input[num_cols] = scaler.transform(X_input[num_cols])

    # 4. Prédiction
    prediction = model.predict(X_input)[0]

    return render_template('predict.html', prediction=round(prediction, 2), inputs=request.form)

@app.route('/download/<filename>')
def download_cleaned_data(filename):
    DOWNLOAD_FOLDER = os.path.join('static', 'downloads')
    return send_from_directory(DOWNLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)