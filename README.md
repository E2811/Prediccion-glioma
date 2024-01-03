# Prediccion-glioma

## Desarrollo de una aplicación web para la predicción del grado de glioma cerebral de un paciente aplicando técnicas de aprendizaje automático sobre unas características clínicas y moleculares

***Máster en bioinformática y bioestadística UOC-UB***
Trabajo de Fin de Master del curso 2023/2024 de Elisa Martín-Montalvo Pérez

### Resumen del Trabajo

El diagnóstico preciso de la gradación de los gliomas es esencial para mejorar la eficacia de los tratamientos y reducir tanto el tiempo necesario para el diagnóstico como los costosos procedimientos de prueba. Históricamente, la clasificación de la agresividad de estos tumores se basaba principalmente en imágenes médicas, sin embargo, recientemente ha aumentado la utilización de características moleculares en este contexto. En este estudio, se ha propuesto una metodología basada en el aprendizaje automático supervisado para clasificar y seleccionar las características más relevantes, tanto moleculares como clínicas. Se ha empleado una serie de algoritmos, el Support Vector Machines (SVM), la Regresión Logística, KNN (Vecinos Cercanos), Random Forests (RF) y Gradient Boosting, diseñando múltiples modelos con diversas configuraciones de parámetros. Para determinar el modelo óptimo, se han evaluado las métricas de f1-score, sensibilidad, precisión y especificidad, dando prioridad a la simplicidad del modelo. El modelo seleccionado como óptimo ha sido SVM con kernel radial, alcanzando un 90% de precisión. Las variables más influyentes en la clasificación han sido la presencia de mutaciones en los genes IDH1, PTEN, ATRX, IDH2 y CIC, junto con la edad al momento del diagnóstico. En contraste, hemos observado que genes como CSMD3, SMARCA4, FAT4 y BCOR no tienen una relevancia significativa en el modelo. Además, se ha implementado una aplicación web interactiva, permitiendo a los usuarios realizar predicciones de forma accesible y amigable.

### Documentos

En el repositorio, se encuentran los siguientes elementos:
 - Carpeta *data*: Contiene varios archivos CSV con datos en bruto utilizados para el entrenamiento y la validación del modelo.
 - Carpeta *src*: Contiene el código fuente principal, que incluye:
    -	*streamlit_app.py*: Desarrollo de la aplicación web.
    -	*models.py*: Funciones para implementar modelos, métricas de evaluación y gráficos de importancia de variables e interpretabilidad.
    -	*EDA_functions.py*: Funciones auxiliares para análisis exploratorio de datos, preprocesamiento y análisis estadísticos.
    -	*Selected_model.py*: Clase con el modelo seleccionado para predicciones, reentrenamiento y archivos guardados necesarios.
    -	*requirements.txt*: Lista de librerías que deben instalarse durante el despliegue de la aplicación.
-	Carpeta *models*: Contiene modelos seleccionados y transformadores de preprocesamiento de datos.
-	Cuaderno *EDA Analysis TFM*: Guía para el análisis descriptivo de los datos, con pasos secuenciales para obtener gráficos relevantes y comprender los datos.
- Cuaderno *Modelización TFM*: Realiza la transformación de datos, división en conjuntos de entrenamiento/prueba e implementación de modelos, evaluando sus métricas.

Podemos acceder a la página web a través del siguiente link

```
https://prediccion-glioma.streamlit.app/
```



