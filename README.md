# Análisis de Sentimientos en Reseñas de Amazon usando Machine Learning y Deep Learning
> ⚠️ Si no puedes visualizar el notebook directamente en GitHub, puedes abrirlo en Google Colab:

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juancaalcaraz/Analisis-de-Sentimientos-en-Reviews-de-Amazon-usando-Machine-Learning-y-Deep-Learning/blob/main/Analisis_de_sentimientos_con_Scikit_learn.ipynb)

---

## 📌 Descripción

Este proyecto aborda la clasificación de reseñas de productos de Amazon en cinco niveles de sentimiento, desde **muy negativo (0)** hasta **muy positivo (4)**. Dado que el conjunto de datos presenta un fuerte desbalance de clases, se implementaron diversas técnicas para mitigar este efecto y mejorar la capacidad de los modelos para predecir correctamente todas las clases, no solo la mayoritaria.

Se comparan modelos de machine learning clásicos con redes neuronales profundas:

- Regresión Logística
- Random Forest
- Multilayer Perceptron (MLP)
- Red Neuronal Bidireccional LSTM (BiLSTM)

---

## 🎯 Objetivos

- Clasificar reseñas de texto en cinco niveles de sentimiento.
- Manejar el desbalance de clases mediante técnicas de muestreo y ajustes en la función de pérdida.
- Comparar modelos tradicionales y redes neuronales para evaluar su desempeño y aplicabilidad en contextos reales.

---

## 📚 Dataset

El conjunto de datos proviene de reseñas de Amazon, y contiene aproximadamente 7.000.000 de muestras etiquetadas en cinco clases. La distribución de clases es altamente desbalanceada:

| Clase | Descripción       | Proporción estimada |
|-------|-------------------|----------------------|
| 0     | Muy negativo      | ~5%                  |
| 1     | Negativo          | ~7%                  |
| 2     | Neutro            | ~10%                 |
| 3     | Positivo          | ~15%                 |
| 4     | Muy positivo      | ~63%                 |

> ⚠️ **Nota:** Este dataset es propiedad de Amazon y se utiliza bajo una licencia de uso académico/no comercial, tal como fue presentado en el artículo:  
> **Ni, J., Li, J., & McAuley, J. (2019).** *Justifying recommendations using distantly-labeled reviews and fine-grained aspects*. EMNLP.  
> Más información: [https://nijianmo.github.io/amazon](https://nijianmo.github.io/amazon)

---

## 🧪 Metodología

- **Preprocesamiento:** limpieza, tokenización, vectorización (HashingVectorizer, TF-IDF).
- **Manejo del desbalance:** técnicas de rebalanceo (SMOTE, undersampling), y funciones de pérdida adaptativas (class weights, focal loss).
- **Modelado:** implementación de cuatro modelos con distintas arquitecturas.
- **Evaluación:** métricas de *precision*, *recall*, *F1-score*, *accuracy*, matrices de confusión y curvas de aprendizaje.

---

## 📈 Resultados

### 🔹 Regresión Logística

| Clase | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.88      | 0.75   | 0.81     | 6734    |
| 1     | 0.27      | 0.38   | 0.32     | 760     |
| 2     | 0.43      | 0.57   | 0.49     | 560     |
| 3     | 0.34      | 0.41   | 0.37     | 1512    |
| 4     | 0.16      | 0.24   | 0.19     | 434     |

- **Accuracy total:** 64%

---

### 🔹 MLP (Multilayer Perceptron)

| Clase | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.78      | 0.90   | 0.84     | 6734    |
| 1     | 0.32      | 0.20   | 0.25     | 760     |
| 2     | 0.55      | 0.40   | 0.46     | 560     |
| 3     | 0.35      | 0.24   | 0.29     | 1512    |
| 4     | 0.21      | 0.18   | 0.20     | 434     |

- **Accuracy total:** 68.5%

---

### 🔹 BiLSTM (Red Neuronal LSTM Bidireccional)

| Clase | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.445     | 0.746  | 0.557    | 1084    |
| 1     | 0.000     | 0.000  | 0.000    | 839     |
| 2     | 0.363     | 0.354  | 0.358    | 1490    |
| 3     | 0.449     | 0.246  | 0.318    | 2996    |
| 4     | 0.840     | 0.932  | 0.884    | 13591   |

- **Accuracy total:** 73.7%
- **F1 macro:** 0.424  
- **F1 ponderado:** 0.705

**Interpretación:**  
Aunque el modelo logra una buena precisión global gracias al excelente desempeño en la clase mayoritaria (clase 4), las clases minoritarias siguen teniendo un rendimiento pobre, especialmente la clase 1, que no fue correctamente predicha en ningún caso. Esto refleja que el modelo, a pesar de su complejidad y capacidad para capturar el contexto, **no logra generalizar bien frente a un dataset tan desbalanceado**, incluso con técnicas adicionales aplicadas.

---

## 📊 Interpretación general

Se observa que todos los modelos tienden a desempeñarse mejor en la clase mayoritaria (muy positivo), mientras que las clases minoritarias presentan menor precisión y recall. Técnicas como **SMOTE** y la estrategia **jerárquica de predicción en dos etapas** (polaridad general → intensidad) fueron implementadas, aunque **no lograron mejoras sostenidas en la distribución real** del conjunto de datos.

Estos resultados reflejan las limitaciones de los enfoques supervisados tradicionales frente a datasets altamente desbalanceados y con clases semánticamente difusas.

---

## 🖼️ Visualizaciones
### Desbalanceo de clases:
Esta es la distribución de las clases en el dataset:

![Desbalance de clases](img/Desbalance.png)

Como se puede observar en el gráfico, la clase **excelente** está sobrerrepresentada, representando más del 60% del total del dataset.

---

### Reporte de `SGDClassifier`:
En el primer entrenamiento se utilizó **SGDClassifier** con `loss='log'` y el vectorizador **HashingVectorizer**.

![SGDClassifier reporte](img/SGDC_reporte.png)

Como se puede ver, las clases **excelente** y **pésima** son las que obtienen mejores resultados.

---

### Regresión logística:
La regresión logística fue otro modelo probado para la clasificación. A continuación se muestra su resultado con la distribución original de los datos:

![Matriz_confusion_LR](img/Matriz_confusion_LR.png)

Aquí se muestra el resultado al entrenar el modelo con datos balanceados manualmente y evaluarlo con la distribución original:

![Matriz_confusion_LR_balanceado](img/Matriz_LR_balanceado_test_desbalanceado.png)

---

También probamos dividir el problema en dos etapas:  
1. Determinar primero la **polaridad general** de la reseña.  
2. Luego clasificarla de forma más específica con otro modelo entrenado para identificar el **nivel de sentimiento**.

**Matriz de polaridad:**

![Matriz_polaridad](img/Matriz_polaridad.png)

**Matriz del umbral positivo con umbral de decisión:**

![Matriz_pos_umbral](img/Matriz_pos_umbral.png)

---

### Curva de pérdida - MLP:
Otro clasificador que probamos fue el **Multilayer Perceptron (MLP)**.

![MLP curva de pérdida](img/Curva_loss_MLP.png)

Como se puede observar, la pérdida va disminuyendo a lo largo de las épocas de entrenamiento, lo que indica un buen ajuste a los datos de entrenamiento.

---

### Bidirectional LSTM:
Esta es la curva de pérdida del entrenamiento y la validación del **Bidirectional LSTM**:

![LSTM curva de pérdida](img/Curva_de_perdida_Bidirectional_LSTM.png)

Se puede ver cómo la pérdida de entrenamiento disminuye con las épocas. Sin embargo, la pérdida de validación no varía demasiado a partir de la **época 4**, lo que indica que el modelo deja de capturar nuevos patrones a partir de ese punto.

---

**Matriz de confusión de Bidirectional LSTM:**

![Matriz de confusión](img/Matriz_confusion_Biderectional.png)

A partir de esta matriz podemos inferir que el modelo clasifica adecuadamente los extremos de los polos del sentimiento, así como el centro. Esto sugiere que, si la clasificación se redujera a solo tres clases (por ejemplo: negativa, neutra y positiva), nuestro **Bidirectional LSTM** podría tener un desempeño significativamente mejor que el actual.
 