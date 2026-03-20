# Detección de Cáncer de Seno con Explainable AI - Estudio de Confianza Médica

## What This Is

Un sistema de inteligencia artificial para detección de cáncer de seno que combina datos clínicos con análisis de mamografías, diseñado específicamente para investigar si las explicaciones generadas por IA (SHAP para datos clínicos, Grad-CAM para imágenes) aumentan la confianza de los médicos radiólogos en las predicciones del modelo. Este proyecto de tesis de maestría busca producir evidencia empírica sobre la utilidad real de la XAI en contextos médicos.

## Core Value

Producir un sistema funcional de detección de cáncer + estudio de validación clínica que demuestre objetivamente si las explicaciones de IA mejoran la toma de decisiones médicas.

## Requirements

### Validated

- ✓ Stack tecnológico definido: Python 3.10+, PyTorch, XGBoost, SHAP, Grad-CAM
- ✓ Arquitectura 2-etapas validada: Triaje clínico (XGBoost) → Análisis de imagen (CNN/ViT) → Fusión multimodal
- ✓ Datasets identificados: CBIS-DDSM (~3,000 imágenes con metadatos clínicos)
- ✓ Pipeline de datos planificado: descarga, limpieza, preprocesamiento DICOM/PNG
- ✓ Módulos de XAI seleccionados: SHAP para features tabulares, Grad-CAM para visión
- ✓ Esquema de abstracciones definido: RiskScorer, VisionBackbone, FusionNetwork, Explainer

### Active

- [ ] **Dataset**: Descargar y preprocesar CBIS-DDSM con metadatos clínicos alineados
- [ ] **Modelo Clínico**: Entrenar XGBoost/MLP para predicción de riesgo basada en datos tabulares (edad, densidad, antecedentes)
- [ ] **Modelo Visión**: Implementar transfer learning con EfficientNet/ViT pre-entrenado para extracción de features de mamografías
- [ ] **Fusión**: Desarrollar mecanismo de late fusion para combinar riesgo clínico + embeddings visuales
- [ ] **XAI Clínico**: Integrar SHAP para explicar contribución de cada factor clínico al riesgo
- [ ] **XAI Visual**: Integrar Grad-CAM para resaltar regiones de la mamografía que influyen en la predicción
- [ ] **Interfaz Web**: Crear interfaz Streamlit para el estudio con médicos (formulario clínico + upload de imagen + visualización de resultados + explicaciones)
- [ ] **Estudio Clínico**: Diseñar y ejecutar estudio con 2-3 radiólogos evaluando 10-20 casos (con/sin explicaciones)
- [ ] **Métricas Estudio**: Medir confianza en predicción, tiempo de decisión, accuracy diagnóstica, preferencia por tipo de explicación
- [ ] **Documentación Tesis**: Redactar capítulos (introducción, marco teórico, metodología, resultados, conclusiones)
- [ ] **Reproducibilidad**: Documentar seeds, requirements.txt, instrucciones de ejecución

### Out of Scope

- Entrenar modelos desde cero (usar transfer learning para ahorrar tiempo y recursos) — limitación de tiempo de 3-6 meses
- Usar datasets privados o requerir aprobación IRB compleja — se usan datasets públicos CBIS-DDSM y CMMD
- Desplegar en producción hospitalaria — es prototipo de investigación académica
- Implementar real-time processing para streaming — batch processing suficiente para estudio
- Sistema multi-usuario con autenticación — single-user local para estudio controlado
- Comparar más de 3 arquitecturas de fusión — enfocarse en late fusion simple + atención
- Incluir otros tipos de cáncer — scope limitado a cáncer de seno
- Análisis de costo-efectividad clínica — fuera del scope de tesis de 3-6 meses

## Context

**Background del tesista:**
- Background médico/biología (no ingeniería), aprendiendo ML para esta tesis
- Acceso potencial a radiólogos para estudio de usabilidad (por confirmar)
- Recursos: GPU modesta o acceso cloud limitado
- Timeline: 3-6 meses para completar

**Estado del campo:**
- Existen muchos papers sobre XAI médica, pero pocos estudios empíricos sobre si médicos realmente confían/usan las explicaciones
- La arquitectura 2-etapas (triaje clínico primero) es lógica médicamente: primero filtrar por riesgo, luego analizar imagen si es necesario
- SHAP y Grad-CAM son métodos estándar bien documentados, no requieren investigar nuevos algoritmos

**Riesgos identificados:**
- No conseguir médicos para estudio → Plan B: análisis quantitativo de explicabilidad sin usuarios
- Limitaciones computacionales → Usar modelos pre-entrenados, batch sizes pequeños, early stopping
- Curva de aprendizado ML → Priorizar componentes existentes (PyTorch, SHAP) sobre implementaciones custom

## Constraints

- **Tiempo:** 3-6 meses máximo — requiere enfoque en transfer learning, no entrenar desde cero
- **Recursos computacionales:** GPU modesta/cloud limitado — requiere modelos eficientes (EfficientNet vs ResNet), batch sizes pequeños
- **Experiencia técnica:** Background médico aprendiendo ML — requiere usar librerías estándar bien documentadas (no código custom complejo)
- **Datos:** Solo datasets públicos disponibles — CBIS-DDSM y CMMD son los principales
- **Validación clínica:** Depende de disponibilidad de médicos — plan B si no se consiguen: análisis de interpretabilidad quantitativo
- **Reproducibilidad:** Debe ser 100% reproducible para comité evaluador — seeds fijos, requirements.txt, notebooks documentados

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Enfoque: Confianza médica en XAI | Aprovecha background médico del tesista, es investigación novel (pocos estudios empíricos), alcanzable en 3-6 meses | — Pending |
| Arquitectura 2-etapas (triaje → imagen) | Más realista clínicamente, reduce carga computacional (no procesar imágenes de pacientes de bajo riesgo), diferenciador respecto a sistemas monolíticos | — Pending |
| Transfer learning en lugar de entrenar desde cero | Ahorra tiempo y recursos computacionales, permite usar arquitecturas probadas, alcanzable en timeline de 3-6 meses | — Pending |
| Datasets públicos (CBIS-DDSM) | No requieren aprobación IRB, disponibles inmediatamente, estándar en literatura de ML médico, suficientes para tesis de maestría | — Pending |
| Métodos XAI estándar (SHAP, Grad-CAM) | Bien documentados, implementaciones robustas disponibles, suficientes para investigar utilidad sin desarrollar nuevos métodos | — Pending |
| Interfaz Streamlit (no web app compleja) | Rápida de desarrollar, suficiente para estudio con médicos, permite enfocar tiempo en modelo + estudio en lugar de frontend complejo | — Pending |

---
*Last updated: 2026-03-20 after project initialization and deep questioning*
