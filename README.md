# 1MTR58 – Laboratorio 2

**Fundamentos y Aplicaciones de Biomecatrónica – PUCP**  
Semestre 2026-2

## Tema
Reconocimiento de patrones aplicado a **Eye Tracking** y **electrooculografía (EOG)**.

## Archivos principales

- `01_EyeTracking.ipynb`: Experiencia 1 – detección ocular, cálculo de EAR, calibración, PERCLOS y pruebas de robustez.
- `02_EOG_Procesamiento.ipynb`: Experiencia 2 – carga, exploración, construcción de EOGh/EOGv, preprocesamiento, segmentación, ventaneo y extracción de características.
- `03_GripperVirtual.py`: HMI para demostrar la relación entre una clase EOG y la acción del gripper virtual.
- `hmi_demo.csv`: archivo de prueba para ejecutar la HMI en modo demo.
- `data/`: archivos EOG utilizados durante la sesión.
- `requirements.txt`: dependencias del laboratorio.

## Preparación del entorno

Para evitar problemas de compatibilidad se proporciona un entorno preparado con **Python 3.12**.

1. Descargue el repositorio.
2. En **Releases**, descargue `python312_lab2.zip`.
3. Descomprímalo dentro de la carpeta principal del repositorio, de modo que quede la carpeta `python/`.
4. Abra el notebook `.ipynb` en Visual Studio Code.
5. Seleccione como intérprete el archivo `python/python.exe`.
6. Seleccione como **kernel** ese mismo entorno.
7. Ejecute la primera celda y verifique el mensaje `✓ Entorno listo. Puede continuar con el laboratorio.`

> El intérprete y el kernel deben apuntar al mismo `python.exe`.

## Trabajo durante la sesión

### Experiencia 1 – Eye Tracking

Se trabajará con detección facial y ocular, cálculo de EAR, calibración del umbral, PERCLOS y evaluación de distintas condiciones de captura.

### Experiencia 2 – EOG

Se trabajará con el sujeto S1 hasta obtener un `features_df` y comparar características entre clases.

Flujo general:

`Carga → exploración → EOGh/EOGv → preprocesamiento → segmentación → ventaneo → características`

## Trabajo para el informe

A partir de los datos de los siete sujetos, cada grupo deberá:

- entrenar y comparar **dos modelos supervisados**;
- reservar un sujeto completo para prueba;
- comparar el desempeño de ambos modelos;
- seleccionar y evaluar el modelo final;
- exportar el modelo y los objetos de preprocesamiento necesarios;
- ejecutar `03_GripperVirtual.py` como demostración de la HMI.

El código desarrollado para el informe debe mantenerse en el repositorio GitHub del grupo.

## HMI

La correspondencia utilizada es:

- Clase 1: saccade outward → **abrir**.
- Clase 2: return saccade → **cerrar**.
- Clase 3: blink → **rotar / confirmar**.

Para ejecutar la demostración:

```bash
python 03_GripperVirtual.py
```

El programa utiliza automáticamente `hmi_demo.csv` para mostrar la correspondencia entre las clases y las acciones del gripper. La integración con un archivo propio de características y el modelo entrenado queda disponible como uso opcional.

## Video del informe

El video debe tener una duración máxima de **2 minutos** y puede ser grabado por **un solo integrante del grupo**, mostrando su pantalla durante la ejecución de la HMI y la evidencia solicitada en el informe.
