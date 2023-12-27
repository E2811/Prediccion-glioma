import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score,f1_score
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest,mutual_info_classif,chi2
from functools import partial
from sklearn.preprocessing import OrdinalEncoder

def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc

def train_test(data,independent_variables,target_variable,categorical_variables,test_size):
    X, y = data[independent_variables], data[target_variable]
    for col in categorical_variables:
        X[col] = X[col].astype('category')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=4)
    return X_train, X_test, y_train, y_test
    
def select_features(X_train, y_train, X_test, k='all'):
    X_train_ = X_train.copy()
    X_test_ = X_test.copy()
    X_train_['Age_at_diagnosis'] = pd.cut(X_train_['Age_at_diagnosis'], [0,10,20,30,40,50,60,70,80,90,100], include_lowest=False).astype('str')
    X_test_['Age_at_diagnosis'] = pd.cut(X_test_['Age_at_diagnosis'], [0,10,20,30,40,50,60,70,80,90,100], include_lowest=False).astype('str')
    
    X_train_enc, X_test_enc =  prepare_inputs(X_train_, X_test_)
    fs = SelectKBest(score_func=chi2, k=k)
    fs.fit(X_train_enc, y_train)
    mask = fs.get_support()
    X_train_fs = X_train.iloc[:,mask]
    X_test_fs = X_test.iloc[:,mask]

    return X_train_fs, X_test_fs, fs

def plot_feature_importance(importance, names, model_type):

    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)


    plt.figure(figsize=(10,8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title('Importancia de las características en el modelo ' + model_type)
    plt.xlabel('Importancia de las características')
    plt.ylabel('Características')
    plt.show()

    return fi_df

def plot_confusion_matrix(ytest,yfit,model_name: str):

  mat = confusion_matrix(ytest, yfit)
  fig, ax = plt.subplots(figsize=(5, 5))
  sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
              # xticklabels=faces.target_names,
              # yticklabels=faces.target_names)
  plt.xlabel('true label')
  plt.ylabel('predicted label')
  plt.title('Matriz de confusión para ' + model_name)
  return plt
  
def roc_curve(y_test, preds_proba, model_name: str):
    fig, ax = plt.subplots(figsize=(6,5))
    fpr, tpr, thresholds = metrics.roc_curve(y_test.values, preds_proba[:, 1], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                              estimator_name=model_name, pos_label=1)
    ax.set_ylabel('Ratio de Verdaderos Positivos')
    ax.set_xlabel('Ratio de Falsos Positivos')
    ax.set_title('Curva ROC para ' + model_name)
    display.plot(ax=ax)
    plt.show()

    return roc_auc, fpr, tpr

#%%
def metrics_resume(y_pred, y_test):

  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred)
  rec = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  auc = roc_auc_score(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred)

  print(acc,prec,rec,f1,auc)

  return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc, 'cm': cm}

def knn_model(X_train, y_train, X_test, y_test,continous_vars, categorical_variables):


    full_pipeline = ColumnTransformer(transformers = [('cat', OneHotEncoder(handle_unknown='ignore'), categorical_variables),
                                                  ('num', StandardScaler(), continous_vars)], remainder='passthrough')


    encoder = full_pipeline.fit(X_train)
    X_train_trf = encoder.transform(X_train)
    X_test_trf = encoder.transform(X_test)

    k_range = list(range(1, 21))
    weight_options = ['uniform', 'distance']
    grid_knn = dict(n_neighbors=k_range, weights=weight_options)

    knn_model =  KNeighborsClassifier()
    CV_knn = GridSearchCV(estimator=knn_model, param_grid=grid_knn, cv = 4, scoring = 'f1')


    CV_knn.fit(X_train_trf, y_train)
    best_knn = CV_knn.best_params_

    print(best_knn)
    model_knn = KNeighborsClassifier(**best_knn)
    model_knn.fit(X_train_trf, y_train)
    y_pred = model_knn.predict(X_test_trf)

    preds_proba = model_knn.predict_proba(X_test_trf)

    roc_auc, fpr, tpr = roc_curve(y_test, preds_proba, model_name='KNN')
    plot_confusion_matrix(y_test, y_pred, model_name = "KNN")

    return model_knn, metrics_resume(y_pred, y_test), roc_auc, fpr, tpr

def svm_model(X_train, y_train, X_test, y_test,continous_vars, categorical_variables,type=''):

    full_pipeline = ColumnTransformer(transformers = [
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_variables),
                                                  ('num', StandardScaler(), continous_vars)], remainder='passthrough')

    encoder = full_pipeline.fit(X_train)
    X_train_trf = encoder.transform(X_train)
    X_test_trf = encoder.transform(X_test)

    with open(f'../models/encoder{type}.pickle', 'wb') as f:
        pickle.dump(encoder, f)

    grid_svm ={'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

    svm_model = svm.SVC()
    CV_svm = GridSearchCV(estimator=svm_model, param_grid=grid_svm, cv = 4, scoring = 'f1')

    CV_svm.fit(X_train_trf, y_train)
    best_svm = CV_svm.best_params_
    print(best_svm)
    model_svm = svm.SVC(**best_svm, probability=True)
    model_svm.fit(X_train_trf, y_train)
    y_pred = model_svm.predict(X_test_trf)

    preds_proba = model_svm.predict_proba(X_test_trf)
    roc_auc, fpr, tpr = roc_curve(y_test, preds_proba, model_name='SVM')
    plot_confusion_matrix(y_test, y_pred, model_name = "SVM")

    y_test.to_excel("../models/ytest.xlsx")
    pd.Series(y_pred).to_excel("../models/ypred.xlsx")
    with open(f'../models/svm{type}.sav', 'wb') as f:
        pickle.dump(model_svm, f)

    return model_svm, metrics_resume(y_pred, y_test), roc_auc, fpr, tpr

def gradient_boosting(X_train, y_train, X_test, y_test):

    grid_gb = dict()
    grid_gb['n_estimators'] = [50, 100]
    grid_gb['learning_rate'] = [1, 0.1, 0.01]
    grid_gb['subsample'] = [0.5, 1.0]
    grid_gb['max_depth'] = [3, 5, 10]
    gb_model = GradientBoostingClassifier()
    CV_gb = GridSearchCV(estimator=gb_model, param_grid=grid_gb, cv = 4, scoring = 'f1')


    CV_gb.fit(X_train, y_train)

    best_gboost = CV_gb.best_params_
    print(best_gboost)

    model_gb = GradientBoostingClassifier(**best_gboost)
    model_gb.fit(X_train, y_train)
    y_pred = model_gb.predict(X_test)

    preds_proba = model_gb.predict_proba(X_test)

    roc_auc, fpr, tpr = roc_curve(y_test, preds_proba, model_name='Gradient Boosting')
    fi_df = plot_feature_importance(model_gb.feature_importances_, X_train.columns, model_type='Gradient Boosting')

    plot_confusion_matrix(y_test, y_pred, model_name = "Gradient Boosting")

    return model_gb,metrics_resume(y_pred, y_test),roc_auc, fpr, tpr

def xgboost_model(X_train, y_train, X_test, y_test):
    constant_params_xgb = {'enable_categorical':True,
                           'objective':'binary:logistic',
                           'tree_method': "hist",
                          'scale_pos_weight':y_train.value_counts()[0]/len(y_train)}

    grid_xgb = {'n_estimators': [100,150],
                'learning_rate': [1, 0.1, 0.01, 0.001],
                'max_depth' : [5,15,20,25],
                'colsample_bytree':[1, 0.7, 0.8, 0.5]
               }

    xgb_model = xgb.XGBClassifier(**constant_params_xgb)
    CV_xgb = GridSearchCV(estimator=xgb_model, param_grid=grid_xgb, cv = 4, scoring = 'f1')


    CV_xgb.fit(X_train, y_train)

    CV_xgb.best_params_.update(constant_params_xgb)
    best_xgboost = CV_xgb.best_params_
    print(best_xgboost)

    model_xgb = xgb.XGBClassifier(**best_xgboost)
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)

    preds_proba = model_xgb.predict_proba(X_test)

    roc_auc, fpr, tpr = roc_curve(y_test, preds_proba, model_name='XGBoost')
    fi_df = plot_feature_importance(model_xgb.feature_importances_, X_train.columns, model_type='XGBoost')

    plot_confusion_matrix(y_test, y_pred, model_name = "XGBoost")

    return model_xgb,metrics_resume(y_pred, y_test),roc_auc, fpr, tpr

def logistic_model(X_train, y_train, X_test, y_test, continous_vars, categorical_variables):

    full_pipeline = ColumnTransformer(transformers = [
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_variables),
                                                  ('num', StandardScaler(), continous_vars)], remainder='passthrough')


    encoder = full_pipeline.fit(X_train)
    X_train_trf = encoder.transform(X_train)
    X_test_trf = encoder.transform(X_test)


    constant_params_lr = {'penalty':'l2', 'class_weight': 'balanced'}
    grid_lr = {'tol': [1e-3, 1e-4, 1e-5, 1e-6],
                'C': [0.1, 1, 10, 100],
                'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}

    lr_model = LogisticRegression(**constant_params_lr)
    CV_lr = GridSearchCV(estimator=lr_model, param_grid=grid_lr, cv = 4, scoring = 'f1')


    CV_lr.fit(X_train_trf, y_train)

    CV_lr.best_params_.update(constant_params_lr)
    best_lr = CV_lr.best_params_
    print(best_lr)

    model_lr = LogisticRegression(**best_lr)

    model_lr.fit(X_train_trf, y_train)
    y_pred = model_lr.predict(X_test_trf)

    preds_proba = model_lr.predict_proba(X_test_trf)

    roc_auc, fpr, tpr = roc_curve(y_test, preds_proba, model_name='Regresión Logistica')
    plot_confusion_matrix(y_test, y_pred, model_name = "Regresión Logistica")

    return model_lr, metrics_resume(y_pred, y_test),roc_auc, fpr, tpr

def random_forest(X_train, y_train, X_test, y_test, continous_vars, categorical_variables):


    param_grid = {
    'n_estimators': [100],
    'max_features': ['sqrt', 'log2', None],
    'max_depth' : [10,20,30],
    'criterion' :['gini', 'entropy', 'log_loss'],
    'oob_score': [True]
    }

    n = len(y_train)
    n_splits = 4
    rfc=RandomForestClassifier(class_weight={0:y_train.value_counts()[0]/n, 1:y_train.value_counts()[1]/n})
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv = n_splits, scoring = 'f1')
    CV_rfc.fit(X_train, y_train)

    rf_best_param = CV_rfc.best_params_
    print(rf_best_param)
    model_rf = RandomForestClassifier(**rf_best_param)
    model_rf.fit(X_train, y_train)

    y_pred = model_rf.predict(X_test)
    preds_proba = model_rf.predict_proba(X_test)

    roc_auc, fpr, tpr = roc_curve(y_test, preds_proba, model_name = "Random Forest")

    plot_feature_importance(model_rf.feature_importances_,X_train.columns, model_type='Random Forest')
    plot_confusion_matrix(y_test, y_pred, model_name = "Random Forest")

    return model_rf, metrics_resume(y_pred, y_test), roc_auc, fpr, tpr

def pca(data,categorical_variables,continous_vars):

  # Entrenamiento modelo PCA con escalado de los datos
  # ==============================================================================
  preprocessor = ColumnTransformer(transformers = [('num', StandardScaler(), continous_vars)], remainder='passthrough')

  pipeline = Pipeline([
      ('preprocessor', preprocessor),
      ('pca', PCA())
  ])
  #pca_pipe = make_pipeline(StandardScaler(), PCA())
  pipeline.fit(data)

  # Se extrae el modelo entrenado del pipeline
  modelo_pca = pipeline.named_steps['pca']

  # Porcentaje de varianza explicada acumulada
  # ==============================================================================
  prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
  print('------------------------------------------')
  print('Porcentaje de varianza explicada acumulada')
  print('------------------------------------------')
  print(prop_varianza_acum)

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
  ax.plot(
      np.arange(len(data.columns)) + 1,
      prop_varianza_acum,
      marker = 'o'
  )

  for x, y in zip(np.arange(len(data.columns)) + 1, prop_varianza_acum):
      label = round(y, 2)
      ax.annotate(
          label,
          (x,y),
          textcoords="offset points",
          xytext=(0,10),
          ha='center'
      )

  ax.set_ylim(0, 1.1)
  ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
  ax.set_title('Porcentaje de varianza explicada acumulada')
  ax.set_xlabel('Componente principal')
  ax.set_ylabel('Por. varianza acumulada')

