# Riesgo de mora e inasistencia (API + interfaz web)

Un backend en FastAPI entrena la regresión logística real a partir de un CSV
y responde predicciones; una página web simple permite cargar datos, consultar a un estudiante y ver el resultado.

## Informe

El problema, el dataset, el modelo, las métricas, la arquitectura, las pruebas y las conclusiones están explicados en el [informe del proyecto](informe.md), ubicado en la raíz del repositorio.

## Qué hace

- **Backend (FastAPI):** recibe el CSV, aplica la limpieza y entrena un modelo
  por evento (mora e inasistencia) con separación temporal entre entrenamiento y prueba, y calcula cada predicción en el momento.
  Valida las entradas y devuelve mensajes de error claros.
- **Frontend (HTML + JavaScript, sin frameworks):** carga del CSV, formulario de consulta, barras con los porcentajes,
  estados de carga, mensajes de error y límites de la solución.

## Requisitos

- Python 3.10 o superior.
- Las librerías de `requirements.txt`.

## Dataset

El CSV de ejemplo está en `data/registro_clases_asistencia.csv`. Se cuenta con la autorización para usar estos datos.

El CSV debe tener estas columnas:

- `Estudiante`
- `Materia / Especialidad`
- `Fecha` (formato `dd/mm/aaaa`)
- `Nro_Clase`
- `Estado_Asistencia`
- `Detalle_Pago_Observaciones`

También puedes usar tu propio CSV desde la interfaz. El sistema lo valida, lo limpia y entrena un modelo nuevo.

## Modelo final

El modelo entrenado se guarda automáticamente en:

```text
models/model.joblib
```

Contiene los dos modelos de regresión logística (mora e inasistencia), el escalador y el historial necesario para predecir.

- La primera vez que ejecutas la aplicación, se entrena con el CSV de `data/` y se crea ese archivo.
- Al reiniciar el servidor, el modelo se carga desde `models/model.joblib`; no se vuelve a entrenar.
- Si subes y entrenas con otro CSV desde la interfaz, el archivo se reemplaza.

### Descargar el modelo final

El modelo no se incluye en el repositorio porque se genera al ejecutar la aplicación.

Puedes obtenerlo de la siguiente manera:

1. **Generarlo localmente** siguiendo la sección *Instalación y ejecución*. Tras abrir la aplicación por primera vez, aparecerá `models/model.joblib`.

Para usar un modelo descargado, colócalo en esta ruta antes de iniciar el servidor:

```text
models/model.joblib
```

## Instalación y ejecución

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Abre <http://127.0.0.1:8000>.

- Si `models/model.joblib` existe, el servidor lo carga y responde de inmediato.
- Si no existe, entrena el modelo con el CSV de ejemplo y lo guarda en `models/model.joblib`.

## Cómo usarlo

1. (Opcional) En **Datos y modelo** elige tu CSV y pulsa *Entrenar con mi CSV*. Se muestran las métricas (Average Precision).
2. En **Consultar un estudiante** escribe el nombre y la fecha de corte, o usa los botones de ejemplo
   (*Caso válido*, *Caso difícil*, *Entrada inválida*), y pulsa *Calcular riesgo*.

## API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/status` | Resumen del modelo actual (estudiantes, fecha de separación, métricas). |
| POST | `/api/train` | Cuerpo: texto del CSV (`Content-Type: text/csv`). Reentrena el modelo y lo guarda. |
| POST | `/api/predict` | Cuerpo JSON `{"student": "...", "date": "dd/mm/aaaa"}`. Devuelve probabilidades y avisos. |

Errores: 400 (CSV inválido o insuficiente), 404 (estudiante inexistente, con sugerencias), 413 (archivo > 2 MB),
422 (fecha o nombre inválidos), 409 (sin modelo entrenado). Documentación interactiva en `/docs`.

Ejemplo:

```bash
curl -X POST http://127.0.0.1:8000/api/predict -H "Content-Type: application/json" \
     -d '{"student": "Marco Antonio", "date": "15/05/2026"}'
```

## Pruebas

```bash
python -m pytest -q
```

- `tests/test_pipeline.py`: lógica del modelo sin servidor (caso válido, caso difícil, entradas inválidas y flujo completo;
  comprueba que las métricas coinciden con el notebook).
- `tests/test_api.py`: lo mismo a través de HTTP, incluyendo el flujo completo *subir CSV, entrenar, consultar*.

| Caso | Entrada | Resultado esperado |
|---|---|---|
| Válido | Marco Antonio, 15/05/2026 | Probabilidades de mora (~20%) e inasistencia (~7%), sin avisos |
| Difícil | Lucas Garrón, 26/02/2026 | Respuesta con aviso de historial insuficiente |
| Inválido | Pedro / fecha `hola` / nombre vacío | Error 404 o 422 con mensaje claro |

Los resultados completos de las pruebas se describen en la sección 6 del [informe](informe.md).

## Estructura

```text
informe.md        informe del proyecto
capturas/         capturas de pantalla usadas en el informe
app/pipeline.py   limpieza, variables, entrenamiento y predicción (lógica del notebook)
app/main.py       API FastAPI
static/index.html interfaz web
tests/            pruebas
data/             CSV de ejemplo
models/           modelo entrenado generado por la aplicación (model.joblib)
```

## Límites

- Son estimaciones con pocos datos (solo 4 faltas reales; la ausencia incluye licencias), no certezas.
- La mora se deduce del texto del detalle; un pago no anotado aparece como mora.
- El modelo apenas supera al azar y en mora pierde contra las reglas simples (ver métricas en la interfaz).
- El modelo es de un solo usuario y se guarda en un archivo local; no está pensado para varios usuarios simultáneos.
- Las fechas mal escritas en el CSV (por ejemplo un año equivocado) se usan tal cual.