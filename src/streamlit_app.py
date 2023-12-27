import streamlit as st 
import datetime
import pandas as pd
from streamlit_option_menu import option_menu
from selected_model import SVM_glioma
from models import plot_confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle
import pathlib
code_dir = pathlib.Path(__file__).parent.resolve()


st.set_page_config(
        page_title="Predicción Glioma",
        page_icon="chart_with_upwards_trend",
    #   layout="wide",
    )

# Sidebar for Navigation
with st.sidebar:
    selected = option_menu('Menu',
    ['Detalles','Predicción de Glioma',
    'Métricas del clasificador'],
    icons=['house','activity','bar-chart-fill'],
    default_index=0)



def detalles():
    st.header("""
    Predicción del grado de glioma
    """,divider='rainbow')  
    st.write("Esta aplicación se ha diseñado con el objetivo de clasificar el glioma cerebral \
    en glioblastoma multiforme o glioma de grado bajo, teniendo en cuenta una serie de características \
    moleculares y clinicas de los pacientes.")

    st.subheader("Uso de la aplicación")
    st.markdown(
    """
    - **Predicción glioma:**\
                Podras subir los datos de los pacientes \
                con el objetivo de obtener la predicción LGG o GBM. 
    - **Métricas del clasificador:** \
                En esta página observarás distintas métricas de evaluación del modelo seleccionado,\
                importancia de las variales y explicabilidad de los modelos. 
    """
    )
    st.subheader("Como realizar la predicción")
    st.markdown(""" :one: **Opción un solo paciente**""")
    st.markdown( """ 
    - Rellenarás los datos del paciente siguiendo las indicaciones de la página. Si marcas el \
    checkbox ***Formulario simple*** la predicción se hará con el modelos de cinco variables. 
    - Una vez completados, pulsa el botón ***Hacer predicción***. 
    - Recibirás el resultado LGG o GBM acorde con la predicción.  
    """)
    st.markdown(""" :two: **Opción un lote de pacientes**""")
    st.markdown(
    """ 
    - Sube un archivo en formato csv separado por comas con los campos correspondientes.
    - Pulsa ***Hacer predicción***
    - Recibirás el resultado en una tabla por paciente con la etiqueta LGG o GBM.  
    """
    )
    st.subheader("Detalles acerca del clasificador")
    st.write("El modelo utilizado en esta aplicación se basa en el algoritmo de Máquinas de Vectores de Soporte (SVM), \
    utilizando un kernel RBF para clasificar el glioma en dos categorías: de grado bajo o glioblastoma multiforme. \
    Este modelo fue entrenado utilizando conjuntos de datos previamente estudiados y disponibles en el repositorio público de UCI Machine Learning Repository.")

    # st.text_area

def  main():  
    st.header("""
    Predicción del grado de glioma
    """,divider='rainbow')    
    #trained_model = pickle.load(open('models/diabetes_trained_model.sav', 'rb'))

    st.subheader('Seleccione de que forma será la entrada de datos')
    opt_pred = st.selectbox("Entrada", ["Un paciente", "Lote de pacientes"])
    if opt_pred == "Un paciente":
        # Mostrar campos del predictor    
        st.subheader('Complete los datos del paciente')
    
        # Diccionario que mapea los valores del selectbox a valores numéricos
        gender_map = {"Masculino": 1,
                    "Femenino": 0
                    }
        race_map = {"negro o africano":1,"blanco":0,'asiático':2,'indio americano o nativo de Alaska':3}

        not_all = st.checkbox('Formulario simple')
        # Crear selectbox para género y convertir a valor numérico
        gen = st.selectbox( "Género:" , options= 
                        ["Masculino" , "Femenino"] 
                        ) 
        gen_val = gender_map[gen] 
        fecha_nacimiento = st.date_input("Fecha de nacimiento",min_value= datetime.date(1910, 1, 1), max_value= datetime.datetime.now())
        fecha_diagnostico = st.date_input("Fecha de diagnostico",min_value=  datetime.date(1910, 1, 1),max_value= datetime.datetime.now())
        edad_diagnostico = round((fecha_diagnostico-fecha_nacimiento).total_seconds()/(24*60*60*365),2)
        if not_all:
            lista_genes = ['IDH1', 'ATRX', 'PTEN', 'CIC']
            genes = st.multiselect(
            "Selecciona los genes alterados de la lista",
            lista_genes,
            )
            genes_selected = [1 if gen in genes else 0 for gen in lista_genes]
            if st.button("Hacer predicción"): 
                x_in =[edad_diagnostico] + genes_selected
                df_x_in = pd.DataFrame([x_in], columns=['Age_at_diagnosis']+lista_genes)
                st.write(df_x_in)
                PATH_MODEL  =code_dir+'models/svm_reduced.sav'
                ENCODER_PATH =code_dir+ 'models/encoder_reduced.pickle'
                svm_clase = SVM_glioma(PATH_MODEL,ENCODER_PATH)
                predict = svm_clase.predict(df_x_in)
                if predict[0] == 0:
                    st.info("El paciente tiene **glioma de bajo grado**")
                else:
                    st.info("El paciente tiene **glioblastoma multiforme**")
        else:
            race =  st.selectbox(
            'Selecciona la raza del paciente',
            options=['negro o africano', 'blanco', 'asiático', 'indio americano o nativo de Alaska'])
            race_val = race_map[race] 
            lista_genes = ['IDH1', 'TP53', 'ATRX',
        'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA', 'NF1', 'PIK3R1', 'FUBP1',
        'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4',
        'PDGFRA']
            genes = st.multiselect(
            "Selecciona los genes alterados de la lista",
            lista_genes,
            )
            genes_selected = [1 if gen in genes else 0 for gen in lista_genes]
            if st.button("Hacer predicción"): 
                x_in =[gen_val,edad_diagnostico,race_val
                    ] + genes_selected
                df_x_in = pd.DataFrame([x_in], columns=['Gender','Age_at_diagnosis', 'Race']+lista_genes)
                st.write(df_x_in)
                PATH_MODEL  ='models/svm.sav'
                ENCODER_PATH = 'models/encoder.pickle'
                svm_clase = SVM_glioma(PATH_MODEL,ENCODER_PATH)
                predict = svm_clase.predict(df_x_in)
                if predict[0] == 0:
                    st.info("El paciente tiene **glioma de bajo grado**")
                else:
                    st.info("El paciente tiene **glioblastoma multiforme**")
    elif opt_pred == "Lote de pacientes":
        uploaded_file = st.file_uploader("Sube un archivo con los datos de los pacientes",
                                    type="csv",
                                    help="El archivo va a ser usado para predecir usando el modelo de ML",
                                    )
        if uploaded_file:
            dataframe = pd.read_csv(uploaded_file)
            st.write(dataframe)
        grade_dic = {0:'LGG',1:'GBM'}
        if st.button("Hacer predicción"): 
            PATH_MODEL  =code_dir+'models/svm.sav'
            ENCODER_PATH = code_dir+'models/encoder.pickle'
            svm_clase = SVM_glioma(PATH_MODEL,ENCODER_PATH)
            predict = svm_clase.predict(dataframe)
            dataframe['Grade'] = pd.Series(predict).replace(grade_dic)
            st.dataframe(dataframe)

def metricas():
    st.header("""Predicción del grado de glioma""",divider='rainbow') 
    target_names = ['LGG','GBM']
    y_true = pd.read_excel(f"{code_dir}mmodels/ytest.xlsx").drop(columns='Unnamed: 0')
    y_pred = pd.read_excel(f"{code_dir}models/ypred.xlsx").drop(columns='Unnamed: 0')
    col1, col2 = st.columns([1, 1])
    col1.write("##")
    col1.write("##")
    figure = plot_confusion_matrix(y_true,y_pred,'SVM')
    col1.dataframe(
    pd.DataFrame(
        classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    ).transpose().round(2))
    col2.pyplot(figure)
    st.divider()
    st.image(f'{code_dir}models/shap_summary.svg', caption='Importancia de variables con Shap')
    st.divider()
    

if selected == 'Detalles':
    detalles()
elif selected == 'Predicción de Glioma':
    main()
elif selected == 'Métricas del clasificador':
    metricas()

# if __name__ == '__main__':
#     main()
