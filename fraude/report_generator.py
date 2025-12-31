from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from datetime import datetime
import numpy as np
import os
import json

def generate_pdf_report(comparison_df, cost_benefit_results, best_threshold, output_path="static/reports/rapport_fraude.pdf"):
    """
    Génère un rapport PDF professionnel de l'analyse de fraude
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Style personnalisé
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=1  # Centré
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#3498db'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # PAGE 1: Titre et Introduction
    story.append(Paragraph("RAPPORT D'ANALYSE", title_style))
    story.append(Paragraph("Système de Détection de Fraude Bancaire", title_style))
    story.append(Spacer(1, 1*cm))
    
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    intro_text = """
    Ce rapport présente les résultats de l'analyse comparative de trois stratégies 
    de détection de fraude bancaire sur des données hautement déséquilibrées (0.5% de fraudes).
    L'objectif est d'optimiser la détection tout en minimisant les coûts opérationnels.
    """
    story.append(Paragraph(intro_text, styles['BodyText']))
    story.append(Spacer(1, 1*cm))
    
    # Résumé exécutif
    story.append(Paragraph("1. RÉSUMÉ EXÉCUTIF", heading_style))
    
    best_strategy = comparison_df.loc[comparison_df['f1'].idxmax(), 'strategie']
    best_f1 = comparison_df['f1'].max()
    best_auc = comparison_df.loc[comparison_df['f1'].idxmax(), 'roc_auc']
    
    summary_text = f"""
    <b>Meilleure stratégie:</b> {best_strategy}<br/>
    <b>F1-Score:</b> {best_f1:.3f}<br/>
    <b>ROC-AUC:</b> {best_auc:.3f}<br/>
    <b>Seuil optimal:</b> {best_threshold:.3f}<br/>
    <b>Bénéfice net estimé:</b> {cost_benefit_results['costs']['net_benefit']:,.2f}€
    """
    story.append(Paragraph(summary_text, styles['BodyText']))
    story.append(Spacer(1, 1*cm))
    
    # PAGE 2: Comparaison des stratégies
    story.append(PageBreak())
    story.append(Paragraph("2. COMPARAISON DES STRATÉGIES", heading_style))
    
    # Tableau comparatif
    table_data = [['Stratégie', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
    for _, row in comparison_df.iterrows():
        table_data.append([
            row['strategie'],
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1']:.3f}",
            f"{row['roc_auc']:.3f}"
        ])
    
    table = Table(table_data, colWidths=[4*cm, 3*cm, 3*cm, 3*cm, 3*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 1*cm))
    
    # Graphique comparaison
    if os.path.exists("static/plots/strategie_comparaison.png"):
        story.append(Image("static/plots/strategie_comparaison.png", width=15*cm, height=7.5*cm))
    
    # PAGE 3: Analyse coût-bénéfice
    story.append(PageBreak())
    story.append(Paragraph("3. ANALYSE COÛT-BÉNÉFICE", heading_style))
    
    cm_data = cost_benefit_results['confusion_matrix']
    cost_text = f"""
    <b>Matrice de Confusion:</b><br/>
    • Vrais Positifs (TP): {cm_data['TP']} fraudes détectées<br/>
    • Faux Positifs (FP): {cm_data['FP']} fausses alertes<br/>
    • Faux Négatifs (FN): {cm_data['FN']} fraudes manquées<br/>
    • Vrais Négatifs (TN): {cm_data['TN']} transactions légitimes<br/><br/>
    
    <b>Impact Financier:</b><br/>
    • Bénéfice total (fraudes évitées): {cost_benefit_results['costs']['benefit_tp_total']:,.2f}€<br/>
    • Coût fausses alertes: {cost_benefit_results['costs']['cost_fp_total']:,.2f}€<br/>
    • Coût fraudes manquées: {cost_benefit_results['costs']['cost_fn_total']:,.2f}€<br/>
    • <b>Bénéfice Net: {cost_benefit_results['costs']['net_benefit']:,.2f}€</b>
    """
    story.append(Paragraph(cost_text, styles['BodyText']))
    story.append(Spacer(1, 0.5*cm))
    
    if os.path.exists("static/plots/cost_benefit_analysis.png"):
        story.append(Image("static/plots/cost_benefit_analysis.png", width=15*cm, height=6*cm))
    
    # PAGE 4: Recommandations
    story.append(PageBreak())
    story.append(Paragraph("4. RECOMMANDATIONS STRATÉGIQUES", heading_style))
    
    reco_text = f"""
    <b>1. Stratégie recommandée:</b> {best_strategy}<br/>
    Cette approche offre le meilleur équilibre entre détection (Recall) et précision.<br/><br/>
    
    <b>2. Seuil de décision optimal:</b> {best_threshold:.3f}<br/>
    Ce seuil maximise le bénéfice net en réduisant les faux positifs coûteux.<br/><br/>
    
    <b>3. Actions prioritaires:</b><br/>
    • Déployer le modèle {best_strategy} en production<br/>
    • Mettre en place un système d'alerte automatique pour les transactions > {best_threshold:.2f}<br/>
    • Former les équipes sur l'interprétation des scores de risque<br/>
    • Monitorer quotidiennement les performances (F1, ROC-AUC)<br/><br/>
    
    <b>4. Améliorations futures:</b><br/>
    • Intégrer des features temporelles (heure, jour de la semaine)<br/>
    • Tester XGBoost/LightGBM pour améliorer les performances<br/>
    • Implémenter un système de feedback pour réentraîner le modèle
    """
    story.append(Paragraph(reco_text, styles['BodyText']))
    
    # Construction du PDF
    doc.build(story)
    print(f"✅ Rapport PDF généré: {output_path}")