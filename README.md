## TFG - Tools to Guarantee Fairness in Machine Learning

Repositorio de mi **trabajo de fin de grado** para el *Doble Grado en Ingeniería Informática y Matemáticas* de la [Universidad de Granada](http://www.ugr.es) sobre un estudio experimental para diferentes modelos de equidad contrafactual. Puede descargar una versión compilada de la memoria en [este enlace](https://github.com/danibolanos/TFG-Guarantee_Fairness_in_ML/releases/download/v1.0.0/TFG_Herramientas_para_Garantizar_Justicia_en_Aprendizaje_Automatico.pdf). Un tutorial para la ejecución del experimento basado en *Jupyter Notebook* puede ser consultado en el [siguiente enlace](https://github.com/danibolanos/TFG-Guarantee_Fairness_in_ML/blob/main/experimentos/tutorial.ipynb). Si desea ejecutarlos en su ordenador será necesario que descargue las dependencias a los paquetes, bien manualmente o bien puede utilizar el siguiente comando:

```
 pip install -r requirements.txt --no-index --find-links file:///tmp/packages
 ```
 
* [PyMC3](https://github.com/pymc-devs/pymc/wiki/Installation-Guide-(Linux)) (opcional)

### Descripción (Español)

Se propone como reto, tras un estudio exhaustivo de la bibliografía y herramientas existentes sobre el tema, realizar un análisis comparativo en *Python* que incluya evaluaciones de equidad de las distintas familias y visualizaciones para la interpretación de resultados y modelos.

Asimismo, en esta tesis se analizarán matemáticamente las definiciones de diversas medidas de equidad y justicia, demostrando sus propiedades, limitaciones y posibles incompatibilidades (teorema de imposibilidad de la equidad), explorando las diferentes opciones de mejora de los resultados obtenidos mediante un proceso de aprendizaje automático en los términos anteriormente propuestos.

Comenzaremos realizando una revisión exhaustiva de las diferentes formalizaciones matemáticas del concepto de equidad existentes en la literatura, analizando las ventajas e inconvenientes de cada una de ellas tanto desde el punto de vista teórico como en lo que respecta a su implementación práctica en problemas concretos de clasificación mediante aprendizaje automático. Además, se realizará un estudio y clasificación de los algoritmos publicados más populares que trabajan con la equidad y se estudiarán las ventajas e inconvenientes de cada uno de ellos.

-----

Repository for my **bachelor's thesis** for the *Double Degree in Computer Engineering and Mathematics* at the [University of Granada](http://www.ugr.es) about an experimental study of different counterfactual fairness models. A compiled version of the notebook can be downloaded at [this link](https://github.com/danibolanos/TFG-Guarantee_Fairness_in_ML/releases/download/v1.0.0/TFG_Herramientas_para_Garantizar_Justicia_en_Aprendizaje_Automatico.pdf). A tutorial for running the experiment based on *Jupyter Notebook* can be found at [this link](https://github.com/danibolanos/TFG-Guarantee_Fairness_in_ML/blob/main/experimentos/tutorial.ipynb). 

### Description (English)

It proposes as a challenge, after an exhaustive study of the existing bibliography and tools on the subject, to carry out a comparative analysis in *Python* that includes fairness evaluations of all the families and visualisations for the interpretation of results and models.

To do so, we will start by carrying out an exhaustive review of the different mathematical formalisations of the concept of fairness in the literature, analysing the advantages and disadvantages of each one both from a theoretical point of view and in terms of their practical implementation in specific classification problems by means of automatic learning. In addition, a study and classification of the most popular published algorithms that work with fairness will be carried out and a study of the advantages and disadvantages of each one will be made.

-----

### Main Bibliography

- Barocas, S., Hardt, M., & Narayanan, A. (2018). Fairness and Machine Learning. fairmlbook. org, 2018. URL: http://www.fairmlbook.org.

- Mitchell, S., Potash, E., Barocas, S., D'Amour, A., & Lum, K. (2018). Prediction-based decisions and fairness: A catalogue of choices, assumptions, and definitions. arXiv preprint arXiv:1811.07867.

- Zafar, M. B., Valera, I., Gomez Rodriguez, M., & Gummadi, K. P. (2017, April). Fairness beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment. In Proceedings of the 26th international conference on world wide web (pp. 1171-1180).

- M. J. Kusner, J. R. Loftus, C. Russell & R. Silva (2018). Counterfactual Fairness.

- Saleiro, P., Kuester, B., Hinkson, L., London, J., Stevens, A., Anisfeld, A., ... & Ghani, R. (2018). Aequitas: A bias and fairness audit toolkit. arXiv preprint arXiv:1811.05577.
