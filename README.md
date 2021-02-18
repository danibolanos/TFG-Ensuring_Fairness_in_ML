# TFG-HGJAA
Repositorio para el Trabajo Fin de Grado: Herramientas para Garantizar Justicia en Aprendizaje Automático.

El aprendizaje automático, al igual que los humanos, puede discriminar y, además, su efecto es más grave por la creencia de que una máquina no tiene prejuicios y por su alta escalabilidad, que magnifica su impacto. Las causas que hacen que el aprendizaje automático no sea justa son diversas, desde el uso de una muestra sesgada o con ejemplos contaminados, hasta la existencia de proxies. 

Existen distintas definiciones de justicia que pueden agruparse en las siguientes familias: 1) paridad demográfica, 2) cuotas igualadas, 3) paridad de tasa predictiva, 4) equidad individual, y 5) equidad contrafactual. Además, algunos de estos grupos son contradictorios entre sí, de modo que conseguir una forma de justica implica empeorar otra. Este asunto de la justicia en aprendizaje automático, por sus importantes implicaciones, está ganando gran relevancia tanto en la comunidad científica como en la empresarial.

El TFG se propone como reto, tras un exhaustivo estudio de la bibliografía y las herramientas existentes sobre el tema, implementar una biblioteca en Python que incluya evaluaciones de justicia de todas las familias y visualizaciones para la interpretación de resultados y modelos.

Asimismo, este TFG analizará matemáticamente las definiciones de diversas medidas de equidad y justicia, demostrando sus propiedades, limitaciones y posibles incompatibilidades, explorando así mismo las diferentes opciones para mejorar los resultados obtenidos mediante un proceso de aprendizaje automático en los términos planteados previamente:  equidad, paridad estadística/demográfica, etc. 

Para ello se tendrá que comenzar realizando una exhaustiva revisión de las diferentes formalizaciones matemáticas del concepto de equidad/justicia en la bibliografía al respecto, analizando las ventajas e inconvenientes de cada una tanto desde el punto de vista teórico como de cara a su implementación práctica en problemas concretos de clasificación mediante aprendizaje automático sin necesidad de reentrenamiento, evitando en la medida de lo posible cualquier tipo de sesgo o trato dispar, por no contar con conjuntos de datos de entrenamiento suficientemente diversos o completos. 

Es bien sabido además que el buen comportamiento de estos modelos de predicción pueden variar significativamente respecto a cierto atributo o cualidad que sea especialmente sensible para un resultado que sea justo e igualitario socialmente, y que esta posible disparidad y falta de equidad se podría expresar en términos de las correspondientes distribuciones de probabilidad, o muestrales, de los conjuntos de datos de entrada y salida para cada uno de estos grupos de individuos potencialmente discriminados, según las variables y resultados considerados. 

##Bibliografía

Barocas, S., Hardt, M., & Narayanan, A. (2017). Fairness in machine learning. NIPS Tutorial, 1

Barocas, S., Hardt, M., & Narayanan, A. (2018). Fairness and Machine Learning. fairmlbook. org, 2018. URL: http://www. fairmlbook. org.

Mitchell, S., Potash, E., Barocas, S., D'Amour, A., & Lum, K. (2018). Prediction-based decisions and fairness: A catalogue of choices, assumptions, and definitions. arXiv preprint arXiv:1811.07867.

Zafar, M. B., Valera, I., Gomez Rodriguez, M., & Gummadi, K. P. (2017, April). Fairness beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment. In Proceedings of the 26th international conference on world wide web (pp. 1171-1180).

Corbett-Davies, S., & Goel, S. (2018). The measure and mismeasure of fairness: A critical review of fair machine learning. arXiv preprint arXiv:1808.00023.

Saleiro, P., Kuester, B., Hinkson, L., London, J., Stevens, A., Anisfeld, A., ... & Ghani, R. (2018). Aequitas: A bias and fairness audit toolkit. arXiv preprint arXiv:1811.05577.
