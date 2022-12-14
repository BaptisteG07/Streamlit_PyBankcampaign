# Importation des librairies

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
# import shap


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

# df = pd.read_csv((bank.csv'), na_values=['.'])
df = pd.read_csv(('C:/Users/asus/Documents/Datascientest/Projet_file_rouge/bank.csv'))

# Preprocessing





#############################################

# Importation des données de colab
#X_train = load('X_train.to_csv')
#X_test = load('X_test.to_csv')
#y_train = load('y_train.to_csv')
#y_test = load('y_test.to_csv')

# Modèle colab
#xgbcl = load('xgbcl_colab.joblib')


# Modèle
#xgbcl = XGBClassifier()
#xgbcl.fit(X_train, y_train)

# Prédiction colab
#y_pred_xgbcl = xgbcl.predict(X_test)

#############################################



################################################################################################

# Supression des variables 'default','duration','day','month' de notre jeu de données
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

####################################################################################################

## Création de la sidebar

st.sidebar.image('logo_datascientest.png')
st.sidebar.markdown("<h1 style='text-align: center; color : green'> PyBankCampaign</h1>", unsafe_allow_html=True)
pages_names = ['Présentation du projet', 'Exploration et visualisation des données', 'Modélisation', 'Vue métier']
page = st.sidebar.radio("Menu",pages_names)
st.sidebar.markdown("<h1 style='text-align: center; color : green'> Projet réalisé par</h1>", unsafe_allow_html=True)
st.sidebar.write('')
st.sidebar.write('BOUNZANGA Joscardin ')
st.sidebar.write('GUILLARD Baptiste')
st.sidebar.write('NGOUSSONG Emmanuel')


## Navigation entre les pages 

if page == 'Présentation du projet' :

    # Insertion des titres
    
    st.markdown("<h1 style='text-align: center; color : orange'> Projet PyBankCampaign</h1>", unsafe_allow_html=True)
    st.write("<h3 style = 'color : skyblue';> Présentation du projet </h3>", unsafe_allow_html = True)
  
    
    # Mise en place de l'image 
    col1, col2, col3=st.columns([1,3,1])
    with col1:
        st.write('')
    with col2:
        st.image('image_streamlit_2.jpg', width=400)
    with col3:
        st.write('')





    # Introduction
    
    st.write(" L’analyse des données marketing est une problématique très classique en sciences des données appliquées dans les entreprises de service.")
    st.write(' Nous disposons d’une base de données, contenant des informations personnelles sur des clients d’une banque qui ont été “télé-marketés”, pour souscrire à un produit que l’on appelle "dépôt à terme".')
    st.write(" Le principe est le suivant, lorsqu’un client souscrit à ce produit, il place une quantité d’argent dans un compte spécifique et ne pourra pas retirer ces fonds avant l’expiration du terme. En échange, le client reçoit des intérêts de la part de la banque à la fin du contrat.")
    
    st.write("<h3 style = 'color : skyblue';> Objectif </h3>", unsafe_allow_html = True)
    
    st.write(" L’objectif de ce projet sera donc de déterminer si un client va adhérer au produit « dépôt à terme », en fonction des résultats obtenus par rapport à la campagne précédente.")
    st.write(" Nous utiliserons des modèles de Machine Learning ainsi que l’interprétabilité de chacun pour illustrer nos analyses.")
















if page == 'Exploration et visualisation des données' :
    
    # Insertion des titres
    st.markdown("<h1 style='text-align: center; color : orange'> Exploration et visualisation des données</h1>", unsafe_allow_html=True)
    
    st.write('Comme dans tout projet de data science, la première étape que nous avons réalisée est une analyse rapide des données.')
    st.write("Ainsi nous avons pu nous approprier le jeu de données mis à notre disposition, émettre quelques hypothèses et y associer des graphiques." )        
    st.write("Notre jeu est constitué des données de 11 162 Clients reparties en 17 variables ")    

    # Affichage des 2 datasets avec checkbox

    st.markdown("<h3 style='color : skyblue'> Dataset</h3>", unsafe_allow_html=True)

    if st.checkbox ('Présentation des variables'):
        st.image('variables.png')
         
    if st.checkbox('Afficher les données'):
        
        liste_datasets = ['Dataset brut', 'Dataset principal']
        dataset = st.radio('Présentation des deux datasets',liste_datasets)
        
        
        if dataset == 'Dataset brut':
            
            n=st.select_slider('Selection de données',options= range(1,100))
            st.dataframe(df.head(n))
            
        if dataset == 'Dataset principal':
            
            st.write("Le nouveau dataframe est obtenu après suppression de default,duration,day et month")
            n=st.select_slider('Selection de données après suppression des variables',options= range(1,100))
            st.dataframe(df0.head(n))




###################################################################################################
#    st.markdown("<h3 style='color : skyblue'> Dataset</h3>", unsafe_allow_html=True)
#
#    if st.checkbox('Afficher les données'):
#        st.image('variables.png')
        
#        liste_datasets = ['Dataset brut', 'Dataset principal']
#        dataset = st.radio('Présentation des deux datasets',liste_datasets)
        
        
#        if dataset == 'Dataset brut':
            
#           st.dataframe(df)
            
#        if dataset == 'Dataset principal':
            
#            df0 = df.drop(['default','duration','day','month'], axis = 1)
#            st.dataframe(df0)
###################################################################################################



    st.markdown("<h3 style='color : skyblue'> Hypothèses</h3>", unsafe_allow_html=True)
        
    # Menu déroulent pour choisir les hypothèses / affichage de l'hypothèse choisi et du graphique

    hypothèses = st.selectbox('Selectionner une hypothèse',('Hypothèse_1', 'Hypothèse_2', 'Hypothèse_3','Hypothèse_4','Hypothèse_5','Hypothèse_6'))
    
    if hypothèses == 'Hypothèse_3':
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

    
    if hypothèses == 'Hypothèse_1':
        st.write("### Ne pas avoir un défaut de crédit favorise-t-il un dépôt à terme ?")
        
        st.write('##### Clients ayant un défaut de crédit par rapport aux clients ayant fait un dépôt à terme')
        fig3 = plt.figure()
        sns.countplot(df.default, hue=df.deposit)
        st.pyplot(fig3)


    if hypothèses == 'Hypothèse_4':
        st.write("### A-t-on moins de chances de faire un dépôt quand on est étudiant et plus de chance d'en faire un en étant retraité ?")
        
        st.write("##### Proportion d'étudiant par rapport à deposit")
        fig4_1 = plt.figure()
        #sns.countplot(df.job=='student', hue=df.deposit)
        sns.countplot(df.job=='student', hue=df.deposit).set_xlabel('Etudiant')
        st.pyplot(fig4_1)
        
        st.write('##### Proportion de retraité par rapport à deposit')
        fig4_2 = plt.figure()
        #sns.countplot(df.job=='retired', hue=df.deposit)
        sns.countplot(df.job=='retired', hue=df.deposit).set_xlabel('Retraité')        
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
        















 
if page == 'Modélisation' :
    
    
    st.markdown("<h1 style='text-align: center;color : orange'> Modélisation</h1>", unsafe_allow_html=True)

    st.markdown("<h3 style='color : skyblue'> Preprocessing</h3>", unsafe_allow_html=True)
    st.write("Nous effectuons l'évaluation sur l'ensemble du dataframe en enlevant les variables 'default', 'duration', 'day' et 'month'")
    st.write('##### Features après preprocessing')

    X_train

    st.markdown('<h5> Tableau récapitulatif des métriques par rapport aux modèles </h5>', unsafe_allow_html=True)

    lst_model = ['LogisticRegression','SVM', 'KNN', 'RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier']
    lst_f1 = [0.63, 0.62, 0.61, 0.63, 0.65, 0.65]
    lst_recall = [0.58, 0.55, 0.58, 0.60, 0.59, 0.59]
    lst_acc = [0.67, 0.67, 0.64, 0.66, 0.69, 0.70]
    
    df_recap_metric = pd.DataFrame(list(zip(lst_model, lst_f1, lst_recall, lst_acc)), columns = ['Modèle', 'f1_score (yes)', 'recall_score (yes)', 'Accuracy'])    
    
    df_recap_metric

    st.write("Nous utiliserons le modèle XGBclassifier car c'est celui avec lequel nous obtenons les meilleurs résultats sur le f1_score et le recall. ")

    # Matrice de confusion
    # cm = pd.crosstab(y_test, y_pred_xgbcl, rownames=['Classe réelle'], colnames=['Classe prédite'])
    
    # Rapport de classification
    rapport_class = classification_report(y_test, y_pred_xgbcl)
    
    
    st.markdown("<h3 style='color : skyblue'> Evaluation</h3>", unsafe_allow_html=True)
    
    # Ckeckbox avec visualisation de la matrice de confusion et du rapport de classification
    if st.checkbox('Afficher les évaluations'):

        liste = ['Matrice de confusion', 'Rapport de classification','Interpretabilité']
        info = st.radio('Matrice de confusion et rapport de classification',liste)
        
        if info == 'Matrice de confusion' :
            st.image('cm_xgbcl.png')
        
        if info == 'Rapport de classification':
            st.image('classification_report_xgbcl.png')
            st.write("Nous obtenons ici un modèle avec un f1_score de 0.65 ce qui peut s'expliquer en partie par le peu d'informations fournies par notre dataframe pour faire les prédictions.")

        if info == 'Interpretabilité':
            # Importance des valeurs selon XGBClassifier
            st.image('feature_importance_xgbcl.png')


    st.markdown("<h3 style='color : skyblue'> Interprétation avec SHAP</h3>", unsafe_allow_html=True)
    if st.checkbox('Afficher les interprétations'):

        st.markdown("<h5/> Importance des variables du XBGC selon les valeurs de Shapley </h5>", unsafe_allow_html=True)
        st.image('Summary_plot_bar_SHAP.png')
        st.image('Summary_plot_SHAP.png')

        st.markdown("<h5/> Interprétation local </h5>", unsafe_allow_html=True)
        st.markdown("<h6/> Client défavorable </h6>", unsafe_allow_html=True)
        st.image('Interprétation_local_no.png')
        st.markdown("<h6/> Client favorable </h6>", unsafe_allow_html=True)
        st.image('Interprétation_local_yes.png')














if page == 'Vue métier' :
    
    
    # Présentation de la vue métier avec les probabilités
    st.markdown("<h1 style='text-align: center;color : orange'> Vue métier</h1>", unsafe_allow_html=True)
    st.write("Nous avons également voulu proposer l'utilisation d'un outil concret.")
    st.write("Tout ceci dans le but d'avoir une application directe de notre projet.")
    st.write("L'objectif est en premier lieu de trouver un seuil de probabilité pouvant ameliorer les performances de notre modèle, ici le 'recall' et le 'f1_score'")
    st.write("Une fois ce seuil obtenu, nous l'utiliserons sur notre modèle finale pour prédire si le banquier doit appeler tel ou tel client pour obtenir une réponse favorable à l'adhésion au contrat à terme.")
    
    if st.checkbox('Afficher les données sur predict_proba'):


        # Fonction de Labelisation
        def to_labels(pos_probs, threshold):
            return (pos_probs >= threshold).astype('int')
        
        #if st.checkbox('Afficher la fonction de labelisation'):
        #    st.write("def to_labels(pos_probs, threshold):")
        #    st.write("return (pos_probs >= threshold).astype('int')")
       
        # Prédiction des probabilités
        prob_reg_xbgc_train = xgbcl.predict_proba(X_train)
    
     
        # Récupération des probabilités de la classe positive
        probs_xgbc = prob_reg_xbgc_train[:,1]
    
        # Définition d'un ensemble de seuil à évaluer
        thresholds = np.arange(0, 1, 0.01)
        
        # Remplacer yes/no par 1/0 pour pouvoir etre exploitable
        y_train_prob_xgbc = y_train.replace(['yes','no'],[1,0])
        
        # Remplacement de yes et no par 1 et 0
        y_test_01 = y_test.replace(['yes','no'],[1,0])

        # Evaluation de chaque seuil
        scores = [f1_score(y_train_prob_xgbc, to_labels(probs_xgbc, t)) for t in thresholds]
        
        # Recherche du meilleur seuil
        ix = np.argmax(scores)
        st.markdown("<h4/> Meilleur seuil de prédiction</h4>", unsafe_allow_html=True)
        # thresholds[ix]
        st.write(0.36) # Valeur prise dans le colab envoyé aux examinateurs

        # F1_score
        st.markdown("<h4/> F1-Score</h4>", unsafe_allow_html=True)
        #scores[ix]
        st.write(0.691) # Valeur prise dans le colab envoyé aux examinateurs
        thresh_max_xgbc = thresholds[ix]
        
        prob_reg_xbgc = xgbcl.predict_proba(X_test)
        y_preds_xbgc = np.where(prob_reg_xbgc[:,1]>thresh_max_xgbc,1,0)
    
        # st.write(f1_score(y_test_01, y_preds_xbgc))

        # Matrice de confusion avec seuil XGBC 
        st.markdown("<h4/> Matrice de confusion avec seuil de probabilité</h4>", unsafe_allow_html=True)
        st.image('Mat_conf.png')
        #cm_prob_xgb = pd.crosstab(y_test_01, y_preds_xbgc, rownames=['Classe réelle'], colnames=['Classe prédite'])
        #cm_prob_xgb
        

        # f1 score XGBClassifier
      #  st.write("f1 score XGBClassifier :")
      #  f1_score_pred = f1_score(y_test_01, y_preds_xbgc)
      #  f1_score_pred
        
        # Rapport de classification
        
        rapport_class_prob = classification_report(y_test_01, y_preds_xbgc)
        st.markdown("<h4/> Rapport de classification avec seuil</h4>", unsafe_allow_html=True)
        st.image('Rap_class_XBGC_proba.png')
        # st.text(rapport_class_prob)


        
    if st.checkbox("Application direct"):

        st.markdown("<h4/> Choix des paramètres</h4>", unsafe_allow_html=True)
        
        age = st.slider(label = "Choix de l'âge", min_value = 18, max_value = 95,step = 1)
        
        balance = st.number_input('Solde du compte',step = 100)
        
        campaign = 1
        
        pdays = -1
        
        previous = -1
        
        job_entrepreneur = 0
        job_housemaid = 0
        job_self_employed = 0
        job_services = 0
        job_unemployed = 0
        job_unknown = 0

        
        job = st.selectbox('Secteur de travail', ['management','blue_collar','technician','admin','retired','student'])
        
        if job == 'admin' :
            job_admin = 1
            job_blue_collar = 0
            job_management = 0
            job_technician = 0
            job_retired = 0
            job_student = 0
        
        if job == 'technician' :
            job_admin = 0
            job_blue_collar = 0
            job_management = 0
            job_technician = 1
            job_retired = 0
            job_student = 0            
            
        if job == 'blue_collar' :
            job_admin = 0
            job_blue_collar = 1
            job_management = 0
            job_technician = 0       
            job_retired = 0
            job_student = 0

        if job == 'management' :
            job_admin = 0
            job_blue_collar = 0
            job_management = 1
            job_technician = 0
            job_retired = 0
            job_student = 0

        if job == 'retired' :
            job_admin = 0
            job_blue_collar = 0
            job_management = 0
            job_technician = 0
            job_retired = 1
            job_student = 0

        if job == 'student' :
            job_admin = 0
            job_blue_collar = 0
            job_management = 0
            job_technician = 0
            job_retired = 0
            job_student = 1



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
        X = [age, balance, campaign, pdays, previous, job_admin, job_blue_collar, job_entrepreneur, job_housemaid, job_management,
             job_retired, job_self_employed, job_services, job_student, job_technician, job_unemployed, job_unknown, marital_divorced,
             marital_married, marital_single, education_primary, education_secondary, education_tertiary, education_unknown, housing_no,
             housing_yes, loan_no, loan_yes, contact_cellular, contact_telephone, contact_unknown, poutcome_failure, poutcome_other,
             poutcome_success, poutcome_unknown]
        
        
        # Redimensionnement de X pour pouvoir l'utiliser dans le modèle
        X = np.array(X).reshape((1,-1))
        
        # Affichage de X avant normalisation
       # st.write("X avant normalisation")
       # st.write(X)
        
        # Normalisation de X
        X = pd.DataFrame(scaler_0.transform(X)) 

        # Affichage de X après normalisation
       # st.write("X après normalisation")
       # st.write(X)        
        
        #Chargement du modèle
        xgbcl = load('xgbcl.joblib')
        
        # Réponse du modèle 
        y_application = xgbcl.predict(X)

        # Affichage de la réponse du modèle
        st.markdown("<h4/> Faut il appeler le client ?</h4>", unsafe_allow_html=True)
        st.write(y_application[0])
        if y_application[0] == 'no':
            st.write('## :-1:')

        if y_application[0] == 'yes':
            st.write('## :+1:')

        # Affichage de la probabilité associée
        st.markdown("<h4/> Probabilité associée</h4>", unsafe_allow_html=True)
        y_prob = xgbcl.predict_proba(X)
        y_prob_yes =  np.delete(y_prob, 0)
        st.write(y_prob_yes[0].round(2))



