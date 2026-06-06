# MarkovCache: Optimización Estocástica de Memoria Caché

Este repositorio contiene un simulador desarrollado desde cero en Python para el diseño, implementación y evaluación de políticas óptimas de gestión y desalojo de memoria caché. El problema de reemplazo de páginas se aborda de manera formal modelándolo como un **Proceso de Decisión de Markov (MDP)**, tomando como base teórica los fundamentos de decisiones secuenciales bajo incertidumbre (basado en Russell & Norvig).

El proyecto contrasta el rendimiento de políticas óptimas derivadas de la teoría de control estocástico frente a algoritmos clásicos de sistemas operativos y enfoques de aprendizaje por refuerzo en línea.

## 🚀 Características Principales

- **Modelado Formal MDP:** Mapeo completo del estado de la caché, peticiones de páginas y penalizaciones por *cache miss* en un entorno de transición estocástica.
- **Solucionadores desde Cero (From Scratch):** Implementación limpia y sin librerías de caja negra de los algoritmos iterativos de Bellman:
  - **Value Iteration** (Iteración de Valores).
  - **Policy Iteration** (Iteración de Políticas).
- **Líneas Base (Baselines) para Comparación:** Evaluación del rendimiento frente a:
  - Heurísticas tradicionales de sistemas operativos: **LRU** (Least Recently Used) y **FIFO** (First-In, First-Out).
  - El límite teórico óptimo: **Algoritmo de Belady** (con conocimiento del futuro).
  - Aprendizaje online sin modelo: Agente **UCB1** (Upper Confidence Bound).
- **Análisis de Sensibilidad:** Pruebas rigurosas sobre el impacto del factor de descuento ($\gamma$), la distribución de las peticiones (e.g., Zipf) y el costo computacional (convergencia).

## 🛠️ Tecnologías Utilizadas

- **Lenguaje:** Python 3.x
- **Librerías principales:** NumPy (procesamiento matricial), Matplotlib / Seaborn (visualización de histogramas y convergencia).

---
*Nota: Este proyecto fue desarrollado originalmente como Trabajo Final para la asignatura de **Modelos Estocásticos** en la **Universidad Nacional de Colombia** (2026-I).*
