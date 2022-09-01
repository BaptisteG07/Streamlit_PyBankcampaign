# Importation des librairies

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap


from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from PIL import Image
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from joblib import dump, load

sns.set_theme()


# Importation du dataset

df = pd.read_csv(('C:/Users/asus/Documents/Datascientest/Projet_file_rouge/bank.csv'), na_values=['.'])

# Preprocessing

df0 = df.drop(['default','duration','day','month'], axis = 1)

# Séparation des features et des targets
target_0 = df0['deposit']
feats_0 = df0.drop('deposit', axis = 1)

# Dichotomisation des features 
feats_0 = pd.get_dummies(feats_0)

# Séparation du jeu de données 
X_train, X_test, y_train, y_test = train_test_split(feats_0,target_0, test_size = 0.2, random_state = 123)

# Standardisation de X_train et X_test
scaler_0 = StandardScaler()
X_train = pd.DataFrame(scaler_0.fit_transform(X_train),columns = X_train.columns)
X_test = pd.DataFrame(scaler_0.transform(X_test),columns = X_test.columns) 

# Entrainement du modèle
xgbcl = XGBClassifier()
xgbcl.fit(X_train, y_train)

# Enregistrement du modèle
dump(xgbcl, 'xgbcl.joblib')

# Prédiction
y_pred_xgbcl = xgbcl.predict(X_test)



## Création de la sidebar

st.sidebar.header('PyBankCampaign')
pages_names = ['Présentation du projet', 'Exploration des données', 'DataVisualisation', 'Modélisation', 'Vue métier']
page = st.sidebar.radio("Menu",pages_names)

## Navigation entre les pages 

if page == 'Présentation du projet' :
    
    # Insertion des titres
    
    st.write("# PyBankCampaign")
    st.write("<h3 style = 'color : blue';> Présentation du projet </h3>", unsafe_allow_html = True)
  
    
    # Mise en place de l'image 

    st.image('Image_streamlit_2.jpg')
    
    # Introduction
    
    st.write(" L’analyse des données marketing est une problématique très classique en sciences des données appliquées dans les entreprises de service.")
    st.write(' Nous disposons d’une base de données, contenant des informations personnelles sur des clients d’une banque qui ont été “télé-marketés”, pour souscrire à un produit que l’on appelle "dépôt à terme".')
    st.write(" Le principe est le suivant, lorsqu’un client souscrit à ce produit, il place une quantité d’argent dans un compte spécifique et ne pourra pas retirer ces fonds avant l’expiration du terme. En échange, le client reçoit des intérêts de la part de la banque à la fin du contrat.")
    
    st.write("<h2 style = 'color : red';> Objectif </h2>", unsafe_allow_html = True)
    
    st.write(" L’objectif de ce projet sera donc de déterminer si un client va adhérer au produit « dépôt à terme », en fonction des résultats obtenus par rapport à la campagne précédente.")
    st.write(" Nous utiliserons des modèles de Machine Learning ainsi que l’interprétabilité de chacun pour illustrer nos analyses.")

    
if page == 'Exploration des données' :
    
    # Insertion des titres
    
    st.write('## Exploration des données')
    
    st.write('Comme dans tout projet de data science, la première étape que nous avons réalisée est une analyse rapide des données.')
    st.write("Ainsi nous avons pu nous approprier le jeu de données mis à notre disposition, émettre quelques hypothèses et y associer des graphiques." )        

    # Affichage des 2 datasets avec checkbox

    st.write("### Dataset")    

    if st.checkbox('Afficher les données'):
        
        liste_datasets = ['Dataset brut', 'Dataset principal']
        dataset = st.radio('Présentation des deux datasets',liste_datasets)
        
        
        if dataset == 'Dataset brut':
            
            st.dataframe(df)
            
        if dataset == 'Dataset principal':
            
            df0 = df.drop(['default','duration','day','month'], axis = 1)
            st.dataframe(df0)
            
    st.write('### Hypothèses')
    
        
    # Menu déroulent pour choisir les hypothèses / affichage de l'hypothèse choisi et du graphique

    hypothèses = st.selectbox('Selectionner une hypothèse',('Hypothèse_1', 'Hypothèse_2', 'Hypothèse_3','Hypothèse_4','Hypothèse_5','Hypothèse_6'))
    
    if hypothèses == 'Hypothèse_1':
        st.write("### Un client a plus de chance de souscrire s'il n'a pas de crédit ?")
        
        st.write('##### Clients ayant un crédit personnel en cours par rapport à deposit')
        fig1 = plt.figure()
        sns.countplot(df.loan, hue = df.deposit)
        st.pyplot(fig1)


    if hypothèses == 'Hypothèse_2':
        st.write("### Être marié favorise-t-il le dépôt ?")
        
        st.write('##### Situation marital par rapport à deposit')
        fig2 = plt.figure()
        sns.countplot(x = df.marital, hue = df.deposit)
        st.pyplot(fig2)

    
    if hypothèses == 'Hypothèse_3':
        st.write("### Ne pas avoir un défaut de crédit favorise-t-il un dépôt à terme ?")
        
        st.write('##### Clients ayant un défaut de crédit par rapport aux clients ayant fait un dépôt à terme')
        fig3 = plt.figure()
        sns.countplot(df.default, hue=df.deposit)
        st.pyplot(fig3)


    if hypothèses == 'Hypothèse_4':
        st.write("### A-t-on moins de chances de faire un dépôt quand on est étudiant et plus de chance d'en faire un en étant retraité ?")
        
        st.write("##### Proportion d'étudiant par rapport à deposit")
        fig4_1 = plt.figure()
        sns.countplot(df.job=='student', hue=df.deposit)
        st.pyplot(fig4_1)
        
        st.write('##### Proportion de retraité par rapport à deposit')
        fig4_2 = plt.figure()
        sns.countplot(df.job=='retired', hue=df.deposit)
        st.pyplot(fig4_2)        


    if hypothèses == 'Hypothèse_5':
        st.write("### Le solde du compte peut-il influencer le choix de faire un dépôt ?")
        
        st.write('##### Répartitions par décile des comptes clients en fonction de deposit')
        df.cut_balance = pd.qcut(df.balance,q = 10)
        fig5 = plt.figure()
        sns.countplot(y = df.cut_balance, hue = df.deposit)
        st.pyplot(fig5)


    if hypothèses == 'Hypothèse_6':
        st.write("### Comment les anciennes campagnes influencent-elles les dépôts ?")

        st.write('##### Client ayant souscrit au produit à la campagne précédente en fontion de deposit')
        fig6 = plt.figure()
        sns.countplot(df.poutcome, hue = df.deposit)
        st.pyplot(fig6)
        
        
        
        
        
        
        


if page == 'DataVisualisation' :
    
    # Affichage de certaines figures 
    
    st.write('## DataVisualisation')
    st.write("#### Nous avons ensuite réalisé quelques visualisations ")
    









    
if page == 'Modélisation' :
    
    st.write('## Modélisation')
    st.write("##### Nous utiliserons le modèle XGBclassifier car c'est celui avec lequel nous obtenons les meilleurs résultats sur le f1_score et le recall. ")
    
    
    
    # Matrice de confusion
    cm = pd.crosstab(y_test, y_pred_xgbcl, rownames=['Classe réelle'], colnames=['Classe prédite'])
    
    # Rapport de classification
    rapport_class = classification_report(y_test, y_pred_xgbcl)
    
    
    st.write("### Evaluation")
    st.write("Nous effectuons l'évaluation sur l'ensemble du dataframe en enlevant les variables 'default', 'duration', 'day' et 'month'")
    
    # Ckeckbox avec visualisation de la matrice de confusion et du rapport de classification
    if st.checkbox('Afficher les données'):
        
        liste = ['Matrice de confusion', 'Rapport de classification']
        info = st.radio('Matrice de confusion et rapport de classification',liste)
        
        if info == 'Matrice de confusion' :
            cm
        
        if info == 'Rapport de classification':
            st.text('Model Report:\n ' + rapport_class)
            st.write("Nous obtenons ici un modèle avec un f1_score de 0.63 ce qui est peu mais au vue de la pauvreté de notre dataframe c'est ce que nous obtenons de mieux.")


if page == 'Vue métier' :
    
    
    
    # Présentation de la vue métier avec les probabilités
    st.write('## Vue métier')
    st.write("#### Nous avons également voulu proposer une vue métier avec l'utilisation de predict_proba.")
    st.write("#### Déterminer un seuil à partir duquel le banquier sera suceptible d'appeler le client pour qu'il fasse un dépôt.")
    
    if st.checkbox('Afficher les données sur predict_proba'):

    
        # Fonction de Labelisation
        def to_labels(pos_probs, threshold):
            return (pos_probs >= threshold).astype('int')
        
        # Remplacement de yes et no par 1 et 0
        y_test_01 = y_test.replace(['yes','no'],[1,0])
    
        
        # Prédiction des probabilités
        prob_reg_xbgc_train = xgbcl.predict_proba(X_train)
    
        # Récupération des probabilités de la classe positive
        probs_xgbc = prob_reg_xbgc_train[:,1]
    
        # Définition d'un ensemble de seuil à évaluer
        thresholds = np.arange(0, 1, 0.01)
        
        # Remplacer yes/no par 1/0 pour pouvoir etre exploitable
        y_train_prob_xgbc = y_train.replace(['yes','no'],[1,0])
        
        # Evaluation de chaque seuil
        scores = [f1_score(y_train_prob_xgbc, to_labels(probs_xgbc, t)) for t in thresholds]
        
        # Recherche du meilleur seuil
        ix = np.argmax(scores)
        st.write("Threshold :")
        thresholds[ix]
        st.write("F-Score :")
        scores[ix]
        thresh_max_xgbc = thresholds[ix]
        
        prob_reg_xbgc = xgbcl.predict_proba(X_test)
        y_preds_xbgc = np.where(prob_reg_xbgc[:,1]>thresh_max_xgbc,1,0)
    
        # Matrice de confusion avec seuil XGBC 
        
        st.write("Matrice de confusion avec seuil de probabilité")
        cm_prob_xgb = pd.crosstab(y_test_01, y_preds_xbgc, rownames=['Classe réelle'], colnames=['Classe prédite'])
    
        cm_prob_xgb
        
        # f1 score XGBClassifier
      #  st.write("f1 score XGBClassifier :")
      #  f1_score_pred = f1_score(y_test_01, y_preds_xbgc)
      #  f1_score_pred
        
        # Rapport de classification
        
        rapport_class_prob = classification_report(y_test_01, y_preds_xbgc)
        st.text('Model Report:\n ' + rapport_class_prob)
    
    if st.checkbox("Application dirècte"):
        
        st.write("#### Choix des paramètres")
        
        age = st.slider(label = "Choix de l'âge", min_value = 18, max_value = 95,step = 1)
        
        balance = st.number_input('Solde du compte',step = 100)
        
        campaign = 1
        
        pdays = -1
        
        previous = -1
        
        job_entrepreneur = 0
        job_housemaid = 0
        job_retired = 0
        job_self_employed = 0
        job_services = 0
        job_student = 0
        job_unemployed = 0
        job_unknown = 0

        
        job = st.selectbox('Secteur de travail', ['management','blue_collar','technician','admin'])
        
        if job == 'admin' :
            job_admin = 1
            job_blue_collar = 0
            job_management = 0
            job_technician = 0
        
        if job == 'technician' :
            job_admin = 0
            job_blue_collar = 0
            job_management = 0
            job_technician = 1
            
        if job == 'blue_collar' :
            job_admin = 0
            job_blue_collar = 1
            job_management = 0
            job_technician = 0       
            
        if job == 'management' :
            job_admin = 0
            job_blue_collar = 0
            job_management = 1
            job_technician = 0           
            
        
        marital = st.selectbox('Situation matrimoniale',['married','single','divorced'])
        
        if marital == 'divorced':
            marital_divorced = 1
            marital_married = 0
            marital_single = 0

        if marital == 'married':
            marital_divorced = 0
            marital_married = 1
            marital_single = 0

        if marital == 'single':
            marital_divorced = 0
            marital_married = 0
            marital_single = 1
            
        education = 'secondary'
        
        if education == 'secondary':
            education_primary = 0
            education_secondary = 1
            education_tertiary = 0
            education_unknown = 0
            
        housing = st.selectbox('Crédit immobilier',['yes','no'])
        
        if housing == 'yes':
            housing_yes = 1
            housing_no = 0
        else :
            housing_yes = 0
            housing_no = 1
           
        loan = st.selectbox('Crédit personnel', ['yes','no'])
        
        if loan == 'yes':
            loan_yes = 1
            loan_no = 0
        else :
            loan_yes = 0
            loan_no = 1
            
        contact_cellular = 1
        contact_telephone = 0
        contact_unknown = 0

        poutcome_failure = 0
        poutcome_other = 0
        poutcome_success = 0    
        poutcome_unknown = 1
        
        
        # Création de la liste des paramètres à renseigner dans le model
        X = [age, balance, marital, housing, loan, job, education, contact, pdays, previous, poutcome, campaign]
        
        # Normalisation de X ???
        
        
        #Chargement du modèle
        xgbcl = load('xgbcl.joblib')
        
        y_application = xgbcl.predict(X)
        
        y_application
        
# Liste des variables 
    
#       'age', 'balance', 'campaign', 'pdays', 'previous', 'job_admin.',
 #      'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
 #      'job_management', 'job_retired', 'job_self-employed', 'job_services',
 #      'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
  #     'marital_divorced', 'marital_married', 'marital_single',
  #     'education_primary', 'education_secondary', 'education_tertiary',
  #     'education_unknown', 'housing_no', 'housing_yes', 'loan_no', 'loan_yes',
  #     'contact_cellular', 'contact_telephone', 'contact_unknown',
  #     'poutcome_failure', 'poutcome_other', 'poutcome_success',
  #     'poutcome_unknown'



