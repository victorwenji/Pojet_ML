from flask import Flask, render_template, request, redirect, url_for
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import joblib
from flask import send_file, send_from_directory
from datetime import datetime
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
from fraude.comparison import compare_all_strategies
from fraude.cost_benefit import analyze_cost_benefit
from fraude.report_generator import generate_pdf_report
from fraude.fraud_reports import generate_fraud_reports   


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

@app.template_filter('datetime')
def format_datetime(value, format='%d/%m/%Y à %H:%M'):
    return datetime.now().strftime(format)

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
    import json
    import joblib
    import shap
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    try:
        # Vérification fichier
        if 'file' not in request.files:
            return render_template('fraude.html', error="Aucun fichier trouvé")

        file = request.files['file']
        if file.filename == '':
            return render_template('fraude.html', error="Aucun fichier sélectionné")
        
        filename = secure_filename(file.filename)

        UPLOAD_FOLDER = os.path.join('uploads')
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        print("\n📂 Chargement des données...")
        df = pd.read_csv(filepath)
        
        if df.empty:
            return render_template('fraude.html', error="Le fichier CSV est vide")
        
        if 'is_fraud' not in df.columns:
            return render_template('fraude.html', error="La colonne 'is_fraud' est manquante")
        
        df_clean = clean_data(df)
        nb_transactions = len(df_clean)
        
        # Analyse exploratoire
        ratio, pie_path, hist_path = analyse_classes(df_clean)
        
        print(f"✅ {nb_transactions} transactions chargées")
        print(f"📊 Ratio fraude: {ratio:.2%}")

        print("\n🔧 Préparation des features...")
        
        # Séparer X et y
        X = df_clean.drop(columns=['is_fraud'])
        y = df_clean['is_fraud']
        
        # Exclusion des colonnes sensibles
        EXCLUDE_COLS = [
            "score_risque_marchand",
            "nb_tentatives_echouees",
            "montant_total_24h",
            "nb_trans_24h"
        ]
        X = X.drop(columns=[c for c in EXCLUDE_COLS if c in X.columns])
        
        # Vérifier qu'il reste des features
        if X.shape[1] == 0:
            return render_template('fraude.html', error="Aucune feature disponible après nettoyage")
        
        # Split train/test avec vérification
        if len(X) < 10:
            return render_template('fraude.html', error="Pas assez de données (minimum 10 transactions)")
        
        # Vérifier qu'il y a au moins quelques fraudes
        fraud_count = y.sum()
        if fraud_count < 2:
            return render_template('fraude.html', 
                                 error=f"Pas assez de fraudes ({fraud_count}). Minimum 2 requises pour l'entraînement.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

        print("\n🔬 Comparaison Baseline vs Class Weight vs SMOTE...")
        
        try:
            comparison_df = compare_all_strategies(X_train, X_test, y_train, y_test)
            print("\n📊 Résultats de comparaison:")
            print(comparison_df)
            
            best_strategy_name = comparison_df.loc[comparison_df['f1'].idxmax(), 'strategie']
            print(f"\n🏆 Meilleure stratégie: {best_strategy_name}")
        except Exception as e:
            print(f"⚠️ Erreur lors de la comparaison: {e}")
            return render_template('fraude.html', 
                                 error=f"Erreur comparaison stratégies: {str(e)}")
            
        print("\n⚙️ Optimisation du seuil de décision...")
        
        try:
            # Charger le meilleur modèle
            best_model = joblib.load("models/smote.pkl")
            
            # Aligner features
            X_test_aligned = X_test.copy()
            if hasattr(best_model, 'feature_names_in_'):
                for col in best_model.feature_names_in_:
                    if col not in X_test_aligned.columns:
                        X_test_aligned[col] = 0
                X_test_aligned = X_test_aligned[best_model.feature_names_in_]
            
            y_prob = best_model.predict_proba(X_test_aligned)[:, 1]
            
            from fraude.threshold_tuning import threshold_tuning
            best_thresh, precisions, recalls, f1_scores, thresholds = threshold_tuning(
                y_test, y_prob, save_path="static/plots/f1_vs_threshold.png"
            )
            
            print(f"✅ Seuil optimal: {best_thresh:.3f}")
            
            # Sauvegarder le seuil
            with open("models/best_threshold.json", "w") as f:
                json.dump({"threshold": float(best_thresh)}, f)
                
        except Exception as e:
            print(f"⚠️ Erreur optimisation seuil: {e}")
            best_thresh = 0.5  # Valeur par défaut

        print("\n💰 Analyse coût-bénéfice...")
        
        try:
            y_pred = (y_prob >= best_thresh).astype(int)
            
            cost_benefit_results = analyze_cost_benefit(
                y_test, y_pred, y_prob, 
                threshold=best_thresh,
                cost_fp=50,
                cost_fn=500,
                benefit_tp=500
            )
            
            print(f"✅ Bénéfice net: {cost_benefit_results['costs']['net_benefit']:,.2f}€")
        except Exception as e:
            print(f"⚠️ Erreur analyse coût-bénéfice: {e}")
            # Valeurs par défaut
            cost_benefit_results = {
                'confusion_matrix': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
                'costs': {
                    'cost_fp_total': 0.0,
                    'cost_fn_total': 0.0,
                    'benefit_tp_total': 0.0,
                    'net_benefit': 0.0
                }
            }

        print("\n🔍 Analyse SHAP (explainabilité)...")
        
        try:
            from fraude.shap_explain import explain
            top_features = explain("models/smote.pkl", X_test)
            print(f"✅ Top features: {top_features[:5]}")
        except Exception as e:
            print(f"⚠️ Erreur SHAP: {e}")
            top_features = X_test.columns.tolist()[:10]

        print("\n📋 Génération du rapport de fraudes...")
        
        try:
            fraud_report_df = generate_fraud_reports(
                "models/smote.pkl", X_test, y_test, y_prob, best_thresh, top_n=10
            )
            print(f"✅ {len(fraud_report_df)} cas de fraude documentés")
        except Exception as e:
            print(f"⚠️ Erreur génération rapport fraudes: {e}")
            import traceback
            traceback.print_exc()
            fraud_report_df = pd.DataFrame()

        print("\n📄 Génération du rapport PDF...")
        
        try:
            generate_pdf_report(
                comparison_df, 
                cost_benefit_results, 
                best_thresh,
                output_path="static/reports/rapport_fraude.pdf"
            )
            print("✅ Rapport PDF généré!")
        except Exception as e:
            print(f"⚠️ Erreur génération PDF: {e}")
            # Continuer sans PDF

        print("\n🎯 Prédictions sur les données uploadées...")
        
        # Aligner features pour prédiction
        X_full_aligned = X.copy()
        if hasattr(best_model, 'feature_names_in_'):
            for col in best_model.feature_names_in_:
                if col not in X_full_aligned.columns:
                    X_full_aligned[col] = 0
            X_full_aligned = X_full_aligned[best_model.feature_names_in_]
        
        y_prob_full = best_model.predict_proba(X_full_aligned)[:, 1]
        y_pred_full = (y_prob_full >= best_thresh).astype(int)
        
        df_results = X.copy()
        df_results["fraud_probability"] = y_prob_full
        df_results["is_fraud_pred"] = y_pred_full
        df_results["score_risque"] = (y_prob_full * 100).round(2)
        
        # Statistiques
        nb_fraudes_detectees = int(df_results["is_fraud_pred"].sum())
        taux_fraude = (nb_fraudes_detectees / nb_transactions * 100) if nb_transactions > 0 else 0
        
        print(f"✅ {nb_fraudes_detectees} fraudes détectées ({taux_fraude:.2f}%)")

        DOWNLOAD_FOLDER = os.path.join('static', 'downloads')
        os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

        result_file = f"results_{filename}"
        df_results.to_csv(os.path.join(DOWNLOAD_FOLDER, result_file), index=False)
        
        print(f"✅ Résultats sauvegardés: {result_file}")
        print("\n" + "="*60)
        print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        print("="*60)
        
        return render_template(
            "fraude_results.html",
            # Statistiques générales
            nb_transactions=nb_transactions,
            nb_fraudes_detectees=nb_fraudes_detectees,
            taux_fraude=f"{taux_fraude:.2f}",
            
            # Comparaison des stratégies
            comparison_table=comparison_df.to_html(classes='table table-striped', index=False),
            best_strategy=best_strategy_name,
            
            # Optimisation du seuil
            best_thresh=f"{best_thresh:.3f}",
            
            # Coût-bénéfice
            cost_benefit=cost_benefit_results,
            
            # Graphiques
            pie_path=url_for('static', filename='plots/class_distribution.png'),
            hist_path=url_for('static', filename='plots/class_histogram.png'),
            comparison_chart=url_for('static', filename='plots/strategie_comparaison.png'),
            radar_chart=url_for('static', filename='plots/strategie_radar.png'),
            threshold_chart=url_for('static', filename='plots/f1_vs_threshold.png'),
            cost_benefit_chart=url_for('static', filename='plots/cost_benefit_analysis.png'),
            confusion_matrix_chart=url_for('static', filename='plots/confusion_matrix_cost.png'),
            shap_summary=url_for('static', filename='plots/shap_summary.png'),
            shap_bar=url_for('static', filename='plots/shap_bar.png'),
            shap_beeswarm=url_for('static', filename='plots/shap_beeswarm.png'),
            
            # Rapports téléchargeables
            result_file=result_file,
            pdf_report="reports/rapport_fraude.pdf",
            fraud_csv="reports/fraud_reports.csv",
            comparison_csv="reports/comparison.csv",
            
            # Tableau des résultats
            table_results=df_results.head(20).to_html(classes='table table-striped table-sm', index=False),
            
            # Top features
            top_features=top_features[:10]
        )
        
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        return render_template('fraude.html', 
                             error=f"Erreur lors du traitement: {str(e)}")
    
@app.route('/upload', methods=['POST'])

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
        df_clean, scaler = feature_engineering(df_raw)
        
        # Sauvegarde du fichier nettoyé
        cleaned_filename = "data_paris_clean.csv"
        cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)
        df_clean.to_csv(cleaned_filepath, index=False)
        
        joblib.dump(scaler, 'models/scaler.joblib')

        # PHASE 3 : Sélection des Top 15 Features
        top_features, df_importance, rf_model, X_train, X_test, y_train, y_test = selection_features(df_clean)
        joblib.dump(top_features, 'models/top_features.joblib')

        # PHASE 4 : Comparaison des modèles
        df_comparatif = entrainer_et_comparer_cv(df_clean[top_features], df_clean['prix'])
        models_html = df_comparatif.to_html(classes='table table-striped table-hover', index=False)
        
        # PHASE 5 : Optimisation
        result_opti = optimiser_modele(X_train[top_features], y_train, X_test[top_features], y_test)
        joblib.dump(result_opti['modele_final'], 'models/modele_paris_final.joblib')


        stats, hist, box, corr_txt, miss, outliers, anomalies, nb = analyser_donnees(df_clean)
    
        # Transmission au template dashboard.html
        return render_template('dashboard.html', 
                                stats=stats, 
                                hist_url=hist, 
                                box_url=box, 
                                corr_txt=corr_txt,
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