## Experimento C — Outdoor vs. Indoor

## Preparación

\%pip install pycolmap plotly open3d pandas tqdm -q
Note: you may need to restart the kernel to use updated packages.

```
# Corrido en consola
# !sudo apt-get update -q
# !sudo apt-get install colmap -y -q
import os, shutil, time, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pycolmap
from pathlib import Path
from datetime import datetime
print("pycolmap:", pycolmap.__version__)
```

pycolmap: 4.0.4

## Paths e inicialización de carpetas

In [36]:

```
BASE_DIR = Path('./expC')
LOGS_DIR = BASE_DIR / 'logs'
RESULTS_DIR = BASE_DIR / 'results'
SPARSE_DIR = BASE_DIR / 'sparse'
DB_DIR = BASE_DIR / 'db'
DATA_DIR = Path('./data')
EXP_D_DIR = Path('./expD')
for d in [LOGS_DIR, RESULTS_DIR, SPARSE_DIR, DB_DIR, EXP_D_DIR]:
    d.mkdir(parents=True, exist_ok=True)
SCENES = ['garden', 'bonsai']
SCENE_TYPE = {'garden': 'outdoor', 'bicycle': 'outdoor',
                'bonsai': 'indoor', 'counter': 'indoor'}
CSV_PATH = RESULTS_DIR / 'df_expC.csv'
SCALE = 4
def get_image_path(scene):
    for scale in [f'images_{SCALE}', 'images']:
        p = DATA_DIR / scene / scale
        if p.exists():
            return p
    raise FileNotFoundError(f"No se encontró carpeta de imágenes para {scene}")
```


## Funciones varias

Para guardar los logs en un txt. Así se monitorea el avance sin ensuciar el cuaderno.

```

In [3]:
def execute_to_file(cmd, log_path, stage_name):
    """Ejecuta un comando y guarda stdout/stderr en el log."""
    print(f" [>] {stage_name}...", end='', flush=True)
    t0 = time.time()
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*20}\nCOMANDO: {cmd}\n{'='*20}\n")
        proc = subprocess.Popen(cmd, shell=True, stdout=f,
            stderr=subprocess.STDOUT, text=True)
        proc.wait()
    dt = time.time() - t0
    status = f"OK ({dt:.1f}s)" if proc.returncode == 0 else f"ERROR (ver {log_path.name})"
    print(f" {status}")
    return proc.returncode, dt
```

Para guardar los resultados en un mismo csv

In [4]:

```
def append_to_csv(metrics, csv_path):
    """Agrega o actualiza una fila en el CSV acumulativo."""
    df_new = pd.DataFrame([metrics])
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=['scene', 'config'], keep='last')
    else:
        df = df_new
    df.to_csv(csv_path, index=False)
    return df
```

Para contar componentes conexas

In [5]:

```
def count_components(output_path):
    """Cuenta el número de componentes conexas generadas por COLMAP."""
    if not output_path.exists():
        return 0
    return len([d for d in output_path.iterdir()
            if d.is_dir() and d.name.isdigit()])
```


## Para correr un pipeline COLMAP

## Algunas definiciones y consideraciones

- reco_path es la ruta al componente conexo de mayor tamaño (el modelo principal) generado por el proceso de Sparse Reconstruction. COLMAP los enumera empezando por el 0 según la cantidad de imágenes registradas.
- objeto reco : carga a memoria RAM toda la información de un modelo 3D generado por COLMAP. Permite conectar las imágenes, cámaras y puntos 3D, calcular métricas de forma fácil usando sus métodos integrados, entre otros.
- reco.num_reg_images() : cuenta cuantas imágenes tiene el modelo 3D. Sirve para evaluar cuantas de las imágenes que se le dieron al modelo fueron usadas.
- reco.compute_num_observations() : calcula la suma de cuantas veces un punto 3D fue vistos por una imagen.
- reco.points3D : los puntos que COLMAP logró reconstruir.
- reco.points3D.values()[x].track : lista de todas las fotos donde aparece ese punto.

Como los archivos son pesados y el tiempo de ejecución es alto, se calcula la mayor cantidad de métricas posibles por ejecución, lo que no significa que se vayan a usar:

- scene, scene_type, config: contexto del epxerimento
- model: cámara usada
- total_images, images_registered, registration_ratio: registro de imágenes
- points_3d: cantidad de puntos reconstruidos
- mean_reprojection_error: promedio de error de ubicación de punto en la imagen y de ubicación caclulada.
- total_observations: cantidad de observaciones de todos los puntos en las imágenes
- obs_per_point: promedio de cámaras que ven a un mismo punto 3D
- avg_track_length: promedio de fotos en las que aparece cada punto
- std_track_length: dispersión de la visibilidad de los puntos
- median_track_length: mediana de de la visibilidad de los puntos
- max_track_length: el número máximo de fotos que comparten un mismo punto 3D
- num_connected_components: número de sub-modelos independientes generados
- execution_time_sec: tiempo de ejecución del proceso, en segundos
- sparse_path: ubicación de los archivos .bin o .txt resultantes
- images_path: ubicación de las fotos originales
- timestamp: fecha y hora de ejecución del pipeline

```
In [6]:
def run_experiment(scene, conf):
    """
    Ejecuta el pipeline COLMAP completo para una escena y configuración.
    Guarda métricas en el CSV acumulativo y retorna el dict de métricas.
    """
    img_path = get_image_path(scene)
    db_path = DB_DIR / f"{scene}_{conf['name']}.db"
    out_path = SPARSE_DIR / scene / conf['name']
    log_path = LOGS_DIR / f"log_{scene}_{conf['name']}.txt"
    out_path.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    if log_path.exists():
        log_path.unlink()
    # Contar imágenes
    imgs_total = len(list(img_path.glob('*.jpg')) +
                list(img_path.glob('*.JPG')) +
                list(img_path.glob('*.png')))
    print(f"\n>>> {scene} | {conf['name']} ({imgs_total} imágenes)")
    t_start = time.time()
    extract_cmd = (
        f"colmap feature_extractor "
        f"--database_path {db_path} "
        f"--image_path {img_path} "
        f"--ImageReader.single_camera 1 "
        f"--ImageReader.camera_model {conf['model']} "
        f"--SiftExtraction.use_gpu 1 "
        f"{conf.get('ext', '')}"
    )
    match_cmd = (
        f"colmap exhaustive_matcher "
        f"--database_path {db_path} "
        f"--SiftMatching.use_gpu 1 "
        f"{conf.get('match', '')}"
    )
    map_cmd = (
        f"colmap mapper "
        f"--database_path {db_path} "
        f"--image_path {img_path} "
        f"--output_path {out_path} "
        f"{conf.get('map', '')}"
    )
    execute_to_file(extract_cmd, log_path, "Extracción de features")
    execute_to_file(match_cmd, log_path, "Matching")
    execute_to_file(map_cmd, log_path, "Mapping (SfM)")
    t_end = time.time()
    reco_path = out_path / "0"
    n_components = count_components(out_path)
    if reco_path.exists():
        reco = pycolmap.Reconstruction(str(reco_path))
        imgs_reg = reco.num_reg_images()
        pts3d = reco.num_points3D()
        n_obs = reco.compute_num_observations()
        tracks = (np.array([p.track.length() for p in reco.points3D.values()])
                if pts3d > 0 else np.array([0]))
        metrics = {
            'scene' : scene,
            'scene_type' : SCENE_TYPE[scene],
            'config' : conf['name'],
            'model' : conf['model'],
            'total_images' : imgs_total,
            'images_registered' : imgs_reg,
            'registration_ratio' : imgs_reg / imgs_total if imgs_total > 0 else 0,
            'points_3d'
            'mean_reprojection_error' : reco.compute_mean_reprojection_error(),
            'total_observations' : n_obs,
            'obs_per_point'
            : n_obs / pts3d if pts3d > 0 else 0,
            : tloat(np.mean(tracks)),
            'median_track_length' : float(np.median(tracks)),
```

```
            'max_track_length' : float(np.max(tracks)),
            'num_connected_components' : n_components,
            'execution_time_sec' : t_end - t_start,
            'sparse_path' : str(reco_path),
            'images_path' : str(img_path),
    }
else:
    print(f" [!] Sin reconstrucción generada para {conf['name']} en {scene}")
    metrics = { : scene,
            'scene' : SCENE_TYPE[scene],
            'config' : conf['name'],
            'model' : conf['model'],
            'total_images' : imgs_total,
            'images_registered' : 0,
            'registration_ratio' : 0.0,
            'points_3d' : 0,
            'mean_reprojection_error' : None,
            'total_observations' : 0,
            'obs_per_point' : 0.0,
            'avg_track_length' : 0.0,
            'std_track_length' : 0.0,
            'median_track_length' : 0.0,
            'max_track_length' : 0.0,
            'num_connected_components' : n_components,
            'execution_time_sec' : t_end - t_start,
            'sparse_path' : None,
            'images_path' : str(img_path),
            'timestamp' : datetime.now().isoformat(),
    }
append_to_csv(metrics, CSV_PATH)
print(f" Tiempo total: {t_end - t_start:.1f}s | "
        f"Registradas: {metrics['images_registered']} | "
    f"Puntos 3D: {metrics['points_3d']}")
return metrics
```


## Verificación de imágenes disponibles

In [7]:

```
images_count = {}
for scene in SCENES:
    ip = get_image_path(scene)
    imgs = (list(ip.glob('*.jpg')) + list(ip.glob('*.JPG')) +
        list(ip.glob('*.png')))
    images_count[scene] = len(imgs)
    print(f"{scene:10s}: {len(imgs):4d} imágenes ({ip})")
garden : 185 imágenes (data/garden/images)
bonsai : 292 imágenes (data/bonsai/images_4)
```


## Configuraciones

Todas corren sobre garden (outdoor) y bonsai (indoor).

## Cam_OpenCV y Ext_HighFeatures

Estas dos configuraciones se eligieron por ser las más informativas para la comparación:

- Cam_OpenCV : modelo de cámara con distorsión radial/tangencial (baseline realista).
- Ext_HighFeatures : extracción con 8000 features - evalúa si más descriptores mejoran la reconstrucción. Inicialmente se intentó con 12000 y 10000 features, pero esto superaba las capacidades de la máquina.

```
In [62]: CONFIGS_EXEC = [
        {
            "name" : "Cam_OpenCV",
            "model" : "OPENCV",
            "ext" : "",
            "match" : "",
        },
            "name" : "Ext_HighFeatures",
            "model" : "OPENCV",
            "ext" : "--SiftExtraction.max_num_features 8000",
            "match" : "",
        },
    ]
    results_exec = []
    for scene in SCENES:
        for conf in CONFIGS_EXEC:
            r = run_experiment(scene, conf)
            results_exec.append(r)
    df_exec = pd.DataFrame(results_exec)
    print("\n=== Resultados ejecutados ===")
    # Algunas de las métricas
    cols_show = ['scene', 'config', 'images_registered', 'registration_ratio',
                'points_3d', 'mean_reprojection_error', 'avg_track_length',
                'num_connected_components', 'execution_time_sec']
print(df_exec[cols_show].to_string(index=False))
```

>>> garden | Cam_OpenCV (185 imágenes)
[>] Extracción de features...

OK (78.3s)
[>] Matching... OK (3365.6s)
[>] Mapping (SfM)... OK (584.0s)
Tiempo total: 4028.1s | Registradas: 185 | Puntos 3D: 146084
>>> garden | Ext_HighFeatures (185 imágenes)
[>] Extracción de features... OK (71.0s)
[>] Matching... OK (3243.2s)
[>] Mapping (SfM)... OK (553.9s)
Tiempo total: 3868.2s | Registradas: 185 | Puntos 3D: 142190
>>> bonsai | Cam_OpenCV (292 imágenes)
[>] Extracción de features... OK (69.9s)
[>] Matching... OK (1783.8s)
[>] Mapping (SfM)... OK (563.2s)
Tiempo total: 2416.9s | Registradas: 292 | Puntos 3D: 100162
>>> bonsai | Ext_HighFeatures (292 imágenes)
[>] Extracción de features... OK (67.8s)
[>] Matching... OK (1723.2s)
[>] Mapping (SfM)... OK (624.1s)
Tiempo total: 2415.1s | Registradas: 292 | Puntos 3D: 98402

=== Resultados ejecutados ===

| scene | images_registered | registration_ratio | points_3d | mean_reprojection_error | avg_track_length | num_connected_component |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| s execution_time_sec |  |  |  |  |  |  |
| garden | 185 | 1.0 | 146084 | 1.235573 | 6.651961 |  |
| 1 |  |  |  |  |  |  |
| garden Ext_HighFeatures | 185 | 1.0 | 142190 | 1.241969 | 6.671397 |  |
| 1 |  |  |  |  |  |  |
| bonsai | 292 | 1.0 | 100162 | 0.377986 | 9.484176 |  |
| 1 |  |  |  |  |  |  |
| bonsai Ext_HighFeatures | 292 | 1.0 | 98402 | 0.380774 | 9.655861 |  |
| 1 |  |  |  |  |  |  |

## Cam_Pinhole

In [63]:

```
CONFIGS_PREV_CAMMODELS = [
    {
        "name" : "Cam_Pinhole",
        "model" : "PINHOLE",
        "ext" : "",
        "match" : "",
        "map" : "",
    },
]
for scene in SCENES:
    for conf in CONFIGS_PREV_CAMMODELS:
        run_experiment(scene, conf)
```

>>> garden | Cam_Pinhole (185 imágenes)
[>] Extracción de features... OK (76.0s)
[>] Matching... OK (3359.8s)
[>] Mapping (SfM)... OK (497.4s)
Tiempo total: 3933.4s | Registradas: 185 | Puntos 3D: 146010
>>> bonsai | Cam_Pinhole (292 imágenes)
[>] Extracción de features... OK (69.9s)
[>] Matching... OK (1828.0s)
[>] Mapping (SfM)... OK (1699.7s)
Tiempo total: 3597.6s | Registradas: 292 | Puntos 3D: 60813

## Ext_LowFeatures

```
In [64]: CONFIGS_PREV_EXT = [
        {
            "name" : "Ext_LowFeatures",
            "model" : "OPENCV",
            "ext" : "--SiftExtraction.max_num_features 4000",
            "match" : "",
            "map" : "",
        },
    ]
    for scene in SCENES:
        for conf in CONFIGS_PREV_EXT:
            run_experiment(scene, conf)
```

>>> garden | Ext_LowFeatures (185 imágenes)
[>] Extracción de features... OK (76.0s)
[>] Matching... OK (852.5s)
[>] Mapping (SfM)... OK (382.3s)
Tiempo total: 1310.8s | Registradas: 185 | Puntos 3D: 81606
>>> bonsai | Ext_LowFeatures (292 imágenes)
[>] Extracción de features... OK (66.2s)
[>] Matching... OK (1533.5s)
[>] Mapping (SfM)... OK (448.4s)
Tiempo total: 2048.1s | Registradas: 292 | Puntos 3D: 94480

## Match_Strict y Match_Loose

In [8]:

```
CONFIGS_PREV_MATCH = [
    {
        "name" : "Match_Strict",
        "model" : "OPENCV",
        "ext" : "",
        "match" : "--SiftMatching.max_error 1.5",
        "map" : "",
    },
    {
        "name" : "Match_Loose",
        "model" : "OPENCV",
        "ext" : "",
        "match" : "--SiftMatching.max_error 6.0",
        "map" : "",
    },
]
for scene in SCENES:
```

```
    for conf in CONFIGS_PREV_MATCH:
        run_experiment(scene, conf)
```

>>> garden | Match_Strict (185 imágenes)
[>] Extracción de features... OK (158.7s)
[>] Matching... OK (3502.5s)
[>] Mapping (SfM)... OK (1061.9s)
Tiempo total: 4723.3s | Registradas: 185 | Puntos 3D: 130468
>>> garden | Match_Loose (185 imágenes)
[>] Extracción de features... OK (140.3s)
[>] Matching... OK (3380.1s)
[>] Mapping (SfM)... OK (546.6s)
Tiempo total: 4067.0s | Registradas: 185 | Puntos 3D: 148290
>>> bonsai | Match_Strict (292 imágenes)
[>] Extracción de features... OK (66.2s)
[>] Matching... OK (1710.6s)
[>] Mapping (SfM)... OK (1145.2s)
Tiempo total: 2922.0s | Registradas: 292 | Puntos 3D: 74536
>>> bonsai | Match_Loose (292 imágenes)
[>] Extracción de features... OK (70.4s)
[>] Matching... OK (1723.4s)
[>] Mapping (SfM)... OK (2144.2s)
Tiempo total: 3938.0s | Registradas: 292 | Puntos 3D: 71672

## Map_Robust

In [9]:

```
CONFIGS_PREV_MAP = [
    \{
        "name" : "Map_Robust",
        "model" : "OPENCV",
        "ext" : "",
        "match" : "",
        "map" : "--Mapper.min_model_size 10",
    \},
]
for scene in SCENES:
    for conf in CONFIGS_PREV_MAP:
        run_experiment(scene, conf)
```

>>> garden | Map_Robust (185 imágenes)
[>] Extracción de features... OK (70.1s)
[>] Matching... OK (3435.4s)
[>] Mapping (SfM)... OK (536.0s)
Tiempo total: 4041.6s | Registradas: 185 | Puntos 3D: 146085
>>> bonsai | Map_Robust (292 imágenes)
[>] Extracción de features... OK (67.7s)
[>] Matching... OK (1687.5s)
[>] Mapping (SfM)... OK (516.4s)
Tiempo total: 2271.7s | Registradas: 10 | Puntos 3D: 1410

## Resultados

```
import pandas as pd
from IPython.display import display, Markdown
```

In [38]:

```
csv_path = f"{RESULTS_DIR}/results_expC.csv"
df = pd.read_csv(csv_path)
display(Markdown("---"))
display(Markdown("### Escena **Bonsai**"))
df_bonsai = df[df['scene'] == 'bonsai']
display(df_bonsai)
display(Markdown("---"))
display(Markdown("### Escena **Garden**"))
df_garden = df[df['scene'] == 'garden']
display(df_garden)
display(Markdown("---"))
display(Markdown("### Ambas escenas"))
display(df)
```

Escena Bonsai
|  | scene | scene_type | config | model | total_images | images_registered | registration_ratio | points_3d | mean_reprojection_error | total_observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2 | bonsai | indoor | Cam_OpenCV | OPENCV | 292 | 292 | 1.000000 | 100162 | 0.377986 | 94995 |
| 3 | bonsai | indoor | Ext_HighFeatures | OPENCV | 292 | 292 | 1.000000 | 98402 | 0.380774 | 95015 |
| 5 | bonsai | indoor | Cam_Pinhole | PINHOLE | 292 | 292 | 1.000000 | 60813 | 1.101698 | 45938 |
| 7 | bonsai | indoor | Ext_LowFeatures | OPENCV | 292 | 292 | 1.000000 | 94480 | 0.382625 | 85077 |
| 10 | bonsai | indoor | Match_Strict | OPENCV | 292 | 292 | 1.000000 | 74536 | 0.372138 | 75230 |
| 11 | bonsai | indoor | Match_Loose | OPENCV | 292 | 292 | 1.000000 | 71672 | 1.148019 | 49272 |
| 13 | bonsai | indoor | Map_Robust | OPENCV | 292 | 10 | 0.034247 | 1410 | 0.688297 | 800 |


## Escena Garden

|  | scene | scene_type | config | model | total_images | images_registered | registration_ratio | points_3d | mean_reprojection_error | total_observatior |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | garden | outdoor | Cam_OpenCV | OPENCV | 185 | 185 | 1.0 | 146084 | 1.235573 | 97174 |
| 1 | garden | outdoor | Ext_HighFeatures | OPENCV | 185 | 185 | 1.0 | 142190 | 1.241969 | 9486C |
| 4 | garden | outdoor | Cam_Pinhole | PINHOLE | 185 | 185 | 1.0 | 146010 | 1.235309 | 97145 |
| 6 | garden | outdoor | Ext_LowFeatures | OPENCV | 185 | 185 | 1.0 | 81606 | 1.325606 | 5125ɛ |
| 8 | garden | outdoor | Match_Strict | OPENCV | 185 | 185 | 1.0 | 130468 | 1.166192 | 90793 |
| 9 | garden | outdoor | Match_Loose | OPENCV | 185 | 185 | 1.0 | 148290 | 1.236637 | 97662 |
| 12 | garden | outdoor | Map_Robust | OPENCV | 185 | 185 | 1.0 | 146085 | 1.235153 | 97175 |


Ambas escenas
|  | scene | scene_type | config | model | total_images | images_registered | registration_ratio | points_3d | mean_reprojection_error | total_observatior |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | garden | outdoor | Cam_OpenCV | OPENCV | 185 | 185 | 1.000000 | 146084 | 1.235573 | 97174 |
| 1 | garden | outdoor | Ext_HighFeatures | OPENCV | 185 | 185 | 1.000000 | 142190 | 1.241969 | 9486C |
| 2 | bonsai | indoor | Cam_OpenCV | OPENCV | 292 | 292 | 1.000000 | 100162 | 0.377986 | 94995 |
| 3 | bonsai | indoor | Ext_HighFeatures | OPENCV | 292 | 292 | 1.000000 | 98402 | 0.380774 | 95015 |
| 4 | garden | outdoor | Cam_Pinhole | PINHOLE | 185 | 185 | 1.000000 | 146010 | 1.235309 | 97145 |
| 5 | bonsai | indoor | Cam_Pinhole | PINHOLE | 292 | 292 | 1.000000 | 60813 | 1.101698 | 4593ɛ |
| 6 | garden | outdoor | Ext_LowFeatures | OPENCV | 185 | 185 | 1.000000 | 81606 | 1.325606 | 5125ɛ |
| 7 | bonsai | indoor | Ext_LowFeatures | OPENCV | 292 | 292 | 1.000000 | 94480 | 0.382625 | 85077 |
| 8 | garden | outdoor | Match_Strict | OPENCV | 185 | 185 | 1.000000 | 130468 | 1.166192 | 90793 |
| 9 | garden | outdoor | Match_Loose | OPENCV | 185 | 185 | 1.000000 | 148290 | 1.236637 | 97662 |
| 10 | bonsai | indoor | Match_Strict | OPENCV | 292 | 292 | 1.000000 | 74536 | 0.372138 | 7523C |
| 11 | bonsai | indoor | Match_Loose | OPENCV | 292 | 292 | 1.000000 | 71672 | 1.148019 | 49272 |
| 12 | garden | outdoor | Map_Robust | OPENCV | 185 | 185 | 1.000000 | 146085 | 1.235153 | 97175 |
| 13 | bonsai | indoor | Map_Robust | OPENCV | 292 | 10 | 0.034247 | 1410 | 0.688297 | 800 |


(a)

```
In [39]:
METRICS = {
    'config': 'Configuración',
    'images_registered': 'Imgs registradas',
    'registration_ratio': 'Ratio registro',
    'points_3d': 'Puntos 3D',
    'mean_reprojection_error': 'Error reproyección (px)',
    'avg_track_length': 'Track length promedio',
    'num_connected_components': 'Componentes conexas',
    'execution_time_sec': 'Tiempo ejecución (s)',
}
df_expC = pd.read_csv(csv_path)
def mostrar_tabla_escena(df, nombre_escena):
    display(Markdown(f"---"))
    display(Markdown(f"### Escena **{nombre_escena.capitalize()}**"))
    filtro = df[df['scene'] == nombre_escena.lower()][list(METRICS.keys())]
    filtro = filtro.rename(columns=METRICS)
    display(filtro)
mostrar_tabla_escena(df_expC, "bonsai")
mostrar_tabla_escena(df_expC, "garden")
```


## Escena Bonsai

|  | Configuración | lmgs registradas | Ratio registro | Puntos 3D | Error reproyección (px) | Track length promedio | Componentes conexas | Tiempo ejecución (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2 | Cam_OpenCV | 292 | 1.000000 | 100162 | 0.377986 | 9.484176 | 1 | 2416.904538 |
| 3 | Ext_HighFeatures | 292 | 1.000000 | 98402 | 0.380774 | 9.655861 | 1 | 2415.084916 |
| 5 | Cam_Pinhole | 292 | 1.000000 | 60813 | 1.101698 | 7.554059 | 1 | 3597.550094 |
| 7 | Ext_LowFeatures | 292 | 1.000000 | 94480 | 0.382625 | 9.004837 | 1 | 2048.094259 |
| 10 | Match_Strict | 292 | 1.000000 | 74536 | 0.372138 | 10.093136 | 1 | 2922.014849 |
| 11 | Match_Loose | 292 | 1.000000 | 71672 | 1.148019 | 6.874651 | 1 | 3938.032947 |
| 13 | Map_Robust | 10 | 0.034247 | 1410 | 0.688297 | 5.675177 | 2 | 2271.660633 |

## Escena Garden

|  | Configuración | Imgs registradas | Ratio registro | Puntos 3D | Error reproyección (px) | Track length promedio | Componentes conexas | Tiempo ejecución (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Cam_OpenCV | 185 | 1.0 | 146084 | 1.235573 | 6.651961 | 1 | 4028.128087 |
| 1 | Ext_HighFeatures | 185 | 1.0 | 142190 | 1.241969 | 6.671397 | 1 | 3868.161248 |
| 4 | Cam_Pinhole | 185 | 1.0 | 146010 | 1.235309 | 6.653318 | 1 | 3933.363019 |
| 6 | Ext_LowFeatures | 185 | 1.0 | 81606 | 1.325606 | 6.281242 | 1 | 1310.753037 |
| 8 | Match_Strict | 185 | 1.0 | 130468 | 1.166192 | 6.959055 | 1 | 4723.318166 |
| 9 | Match_Loose | 185 | 1.0 | 148290 | 1.236637 | 6.585886 | 1 | 4067.003433 |
| 12 | Map_Robust | 185 | 1.0 | 146085 | 1.235153 | 6.651997 | 1 | 4041.641545 |

(b)
\%pip install seaborn -q
Note: you may need to restart the kernel to use updated packages.
import seaborn as sns
markers = \{"bonsai": "s", "garden": "o"\}

In [18]:
df_plot = df_expC[['scene', 'config', 'registration_ratio']].copy()
df_plot['registration_ratio'] = pd.to_numeric(df_plot['registration_ratio'], errors='coerce')
df_plot = df_plot.dropna(subset=['registration_ratio'])
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bar_plot = sns.barplot(
data=df_plot,
x='scene',
y='registration_ratio',
hue='config',
palette='viridis'
)
plt.title('Eficiencia de Registro por Escena y Configuración', fontsize=15)
plt.xlabel('Escena', fontsize=12)
plt.ylabel('Ratio de Registro (Imgs Registradas / Total)', fontsize=12)
plt.ylim(0, 1.1)
for container in bar_plot.containers:
bar_plot.bar_label(container, fmt='\%.2f', padding=3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Configuraciones')
plt.tight_layout()
plt.show()

Eficiencia de Registro por Escena y Configuración
![](https://cdn.mathpix.com/cropped/3956f0d0-b2ec-4100-bcc8-2aae4e526080-07.jpg?height=788&width=1694&top_left_y=2321&top_left_x=270)

In [20]:

```
markers = {"bonsai": "s", "garden": "o"}
# Filtramos y limpiamos de forma segura
df_viz = df_expC.copy()
df_viz['points_3d'] = pd.to_numeric(df_viz['points_3d'], errors='coerce')
df_viz['mean_reprojection_error'] = pd.to_numeric(df_viz['mean_reprojection_error'], errors='coerce')
df_viz = df_viz.dropna(subset=['points_3d', 'mean_reprojection_error'])
```

```
# 2. Crear el gráfico combinado
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")
scatter = sns.scatterplot(
    data=df_viz,
    x='points_3d',
    y='mean_reprojection_error',
    hue='config', # Color por Configuración
    style='scene', # Forma por Escena
    markers=markers,
    s=250,
    palette='viridis' # Paleta profesional
)
plt.title('Comparación de Escenas y Configuraciones (Precisión vs Densidad)', fontsize=15)
plt.xlabel('Cantidad de Puntos 3D', fontsize=12)
plt.ylabel('Error de Reproyección (px)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Config y Escena')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# 3. Mostrar la tabla con los datos especificos y Timestamp
display(Markdown("### \ Datos Detallados de la Comparación"))
# Seleccionamos las columnas de interés incluyendo el timestamp
cols_interes = [
    'scene', 'config', 'points_3d', 'mean_reprojection_error',
    'avg_track_length', 'execution_time_sec', 'timestamp'
]
# Mostramos la tabla formateada
display(df_viz[cols_interes].sort_values(['scene', 'config']))
```

![](https://cdn.mathpix.com/cropped/3956f0d0-b2ec-4100-bcc8-2aae4e526080-08.jpg?height=966&width=1341&top_left_y=1167&top_left_x=273)

```
Config y Escena config
```

![](https://cdn.mathpix.com/cropped/3956f0d0-b2ec-4100-bcc8-2aae4e526080-08.jpg?height=42&width=212&top_left_y=1296&top_left_x=1689)

```
Ext_HighFeatures Cam_Pinhole Ext_LowFeatures Match_Strict Match_Loose Map_Robust scene garden bonsai
```

Datos Detallados de la Comparación
|  | scene | config | points_3d | mean_reprojection_error | avg_track_length | execution_time_sec | timestamp |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2 | bonsai | Cam_OpenCV | 100162 | 0.377986 | 9.484176 | 2416.904538 | 2026-05-13T03:32:23.881410 |
| 5 | bonsai | Cam_Pinhole | 60813 | 1.101698 | 7.554059 | 3597.550094 | 2026-05-13T06:18:12.408513 |
| 3 | bonsai | Ext_HighFeatures | 98402 | 0.380774 | 9.655861 | 2415.084916 | 2026-05-13T04:12:39.342548 |
| 7 | bonsai | Ext_LowFeatures | 94480 | 0.382625 | 9.004837 | 2048.094259 | 2026-05-13T07:14:12.375666 |
| 13 | bonsai | Map_Robust | 1410 | 0.688297 | 5.675177 | 2271.660633 | 2026-05-13T15:33:38.380554 |
| 11 | bonsai | Match_Loose | 71672 | 1.148019 | 6.874651 | 3938.032947 | 2026-05-13T13:48:23.440997 |
| 10 | bonsai | Match_Strict | 74536 | 0.372138 | 10.093136 | 2922.014849 | 2026-05-13T12:42:44.607409 |
| 0 | garden | Cam_OpenCV | 146084 | 1.235573 | 6.651961 | 4028.128087 | 2026-05-13T01:47:35.974536 |
| 4 | garden | Cam_Pinhole | 146010 | 1.235309 | 6.653318 | 3933.363019 | 2026-05-13T05:18:14.402904 |
| 1 | garden | Ext_HighFeatures | 142190 | 1.241969 | 6.671397 | 3868.161248 | 2026-05-13T02:52:06.309596 |
| 6 | garden | Ext_LowFeatures | 81606 | 1.325606 | 6.281242 | 1310.753037 | 2026-05-13T06:40:03.607981 |
| 12 | garden | Map_Robust | 146085 | 1.235153 | 6.651997 | 4041.641545 | 2026-05-13T14:55:46.515197 |
| 9 | garden | Match_Loose | 148290 | 1.236637 | 6.585886 | 4067.003433 | 2026-05-13T11:54:01.284377 |
| 8 | garden | Match_Strict | 130468 | 1.166192 | 6.959055 | 4723.318166 | 2026-05-13T10:46:12.862758 |


In [22]:

```
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_expC,
    x='config',
    y='avg_track_length',
    hue='scene',
    palette={'bonsai': '#1f77b4', 'garden': '#ff7f0e'}
)
plt.title('Robustez: Track Length Promedio (Bonsai vs Garden)', fontsize=15)
plt.ylabel('Promedio de fotos por punto')
```

```
plt.xlabel('Configuración')
plt.legend(title='Escena')
plt.grid(axis='y', alpha=0.3)
plt.show()
```

Robustez: Track Length Promedio (Bonsai vs Garden)
![](https://cdn.mathpix.com/cropped/3956f0d0-b2ec-4100-bcc8-2aae4e526080-09.jpg?height=876&width=1663&top_left_y=333&top_left_x=276)

In [23]:

```
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df_expC,
    x='config',
    y='avg_track_length',
    hue='config',
    style='scene',
    markers=markers,
    s=200,
    legend=True
)
plt.title('Robustez del Modelo: Longitud Promedio de Track', fontsize=15)
plt.xticks(rotation=45)
plt.ylabel('Promedio de fotos que ven cada punto', fontsize=12)
plt.xlabel('Configuración de Experimento', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

Robustez del Modelo: Longitud Promedio de Track
![](https://cdn.mathpix.com/cropped/3956f0d0-b2ec-4100-bcc8-2aae4e526080-09.jpg?height=931&width=1691&top_left_y=1905&top_left_x=273)

In [26]:

```
# Configuraciones a comparar
configs_to_show = ["Cam_OpenCV", "Ext_HighFeatures", "Cam_Pinhole", "Ext_LowFeatures", "Match_Strict", "Match_Loose", "Map_Robust"]
palette = {'bonsai': '#1f77b4', 'garden': '#ff7f0e'}
fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True)
axes = axes.flatten() # Flatten para acceso más fácil
for i, cfg_name in enumerate(configs_to_show):
    # Filtrar el dataframe por la configuración actual
    df_cfg = df_expC[df_expC['config'] == cfg_name]
    for _, row in df_cfg.iterrows():
        path = row['sparse_path']
        if path and pycolmap.Reconstruction(path):
            reco = pycolmap.Reconstruction(path)
            # Obtenemos la longitud de los tracks para cada punto 3D
            lengths = [p.track.length() for p in reco.points3D.values()]
```

```
    # Graficar la distribución
    sns.kdeplot(
        lengths,
        ax=axes[i],
        label=row['scene'],
        fill=True,
        color=palette.get(row['scene']),
        bw_adjust=1.5
    )
axes[i].set_title(f'Configuración: {cfg_name}', fontsize=14)
axes[i].set_xlabel('Fotos en las que aparece un punto (Track Length)')
axes[i].set_xlim(2, 15) # Zoom en el área de mayor interés
axes[i].legend()
# Ocultar los ejes vacíos (8 y 9)
for j in range(len(configs_to_show), len(axes)):
    axes[j].set_visible(False)
axes[0].set_ylabel('Densidad de Puntos')
plt.suptitle('Distribución de Robustez: Bonsai vs Garden', fontsize=16, y=0.995)
plt.tight_layout()
plt.show()
```

Distribución de Robustez: Bonsai vs Garden
![](https://cdn.mathpix.com/cropped/3956f0d0-b2ec-4100-bcc8-2aae4e526080-10.jpg?height=1311&width=1694&top_left_y=880&top_left_x=270)

In [34]:

```
# --- GRILLA 1: MÉTRICAS NORMALIZADAS (2x2 con 3 subplots) ---
fig1 = plt.figure(figsize=(16, 12))
fig1.suptitle('Grilla 1: Eficiencia de Reconstrucción (Normalizada por Fotos Totales)', fontsize=16, fontweight='bold')
# 1. Yield de Puntos 3D (Rendimiento Geométrico)
ax1 = fig1.add_subplot(2, 2, 1)
sns.scatterplot(data=df_expC, x='execution_time_sec', y=df_expC['points_3d']/df_expC['total_images'],
    hue='config', style='scene_type', markers=markers, s=250, ax=ax1, legend=False)
ax1.set_title('Yield de Puntos (points_3d / total_images)')
ax1.set_ylabel('Puntos 3D por Foto de Entrada')
ax1.set_xlabel('Tiempo Total (seg)')
# 2. Densidad de Observaciones (Rendimiento de Correspondencias)
ax2 = fig1.add_subplot(2, 2, 2)
sns.scatterplot(data=df_expC, x='execution_time_sec', y=df_expC['total_observations']/df_expC['total_images'],
    hue='config', style='scene_type', markers=markers, s=250, ax=ax2, legend=False)
ax2.set_title('Densidad de Observaciones (total_observations / total_images)')
ax2.set_ylabel('Obs por Foto de Entrada')
ax2.set_xlabel('Tiempo Total (seg)')
# 3. Costo Unitario (Esfuerzo por imagen)
ax3 = fig1.add_subplot(2, 2, 3)
sns.scatterplot(data=df_expC, x='execution_time_sec', y=df_expC['execution_time_sec']/df_expC['total_images'],
    hue='config', style='scene_type', markers=markers, s=250, ax=ax3, legend=True)
ax3.set_title('Costo Unitario (execution_time_sec / total_images)')
ax3.set_ylabel('Segundos por Foto de Entrada')
ax3.set_xlabel('Tiempo Total (seg)')
ax3.legend(bbox_to_anchor=(1.15, 1), loc='upper left', title="Config & Escena")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# --- GRILLA 2: MAGNITUDES ABSOLUTAS (2x2) ---
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Grilla 2: Magnitudes Absolutas vs Tiempo de Ejecución', fontsize=16, fontweight='bold')
# 1. Puntos 3D Totales
sns.scatterplot(data=df_expC, x='execution_time_sec', y='points_3d', hue='config', style='scene_type',
    markers=markers, s=250, ax=axes2[0, 0], legend=False)
axes2[0, 0].set_title('Total Puntos 3D vs Tiempo')
```

```
# 2. Total Observaciones
sns.scatterplot(data=df_expC, x='execution_time_sec', y='total_observations', hue='config', style='scene_type',
    markers=markers, s=250, ax=axes2[0, 1], legend=False)
axes2[0, 1].set_title('Total Observaciones vs Tiempo')
# 3. Error de Reproyección (Precisión Final)
sns.scatterplot(data=df_expC, x='execution_time_sec', y='mean_reprojection_error', hue='config', style='scene_type',
    markers=markers, s=250, ax=axes2[1, 0], legend=False)
axes2[1, 0].set_title('Precisión (Error Reproyección px) vs Tiempo')
# 4. Ratio de Registro (Éxito del Reconstrucción)
sns.scatterplot(data=df_expC, x='execution_time_sec', y='registration_ratio', hue='config', style='scene_type',
    markers=markers, s=250, ax=axes2[1, 1], legend=True)
axes2[1, 1].set_title('Tasa de Éxito (Registration Ratio) vs Tiempo')
axes2[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

Grilla 1: Eficiencia de Reconstrucción (Normalizada por Fotos Totales)
![](https://cdn.mathpix.com/cropped/3956f0d0-b2ec-4100-bcc8-2aae4e526080-11.jpg?height=1133&width=1699&top_left_y=773&top_left_x=265)

Grilla 2: Magnitudes Absolutas vs Tiempo de Ejecución

![](https://cdn.mathpix.com/cropped/3956f0d0-b2ec-4100-bcc8-2aae4e526080-11.jpg?height=1136&width=1697&top_left_y=2001&top_left_x=267)
(c)

## Exportar datos para Experimento D

```
# Guardar paths de reconstrucciones y métricas base en ./expD/
exp_d_recs = df_expC[['scene', 'config', 'model',
            'sparse_path', 'images_path',
                'images_registered', 'registration_ratio',
                'points_3d', 'mean_reprojection_error',
            'avg_track_length', 'execution_time_sec']].copy()
out_path_d = EXP_D_DIR / 'reconstructions_for_D.csv'
exp_d_recs.to_csv(out_path_d, index=False)
print(f"Guardado para D: {out_path_d}")
print(exp_d_recs[['scene','config','sparse_path']].to_string(index=False))
```

Guardado para D: expD/reconstructions_for_D.csv
scene
sparse path
garden Cam_OpenCV expC/sparse/garden/Cam_OpenCV/0
garden Ext_HighFeatures expC/sparse/garden/Ext_HighFeatures/0
bonsai Cam_OpenCV expC/sparse/bonsai/Cam_OpenCV/0
bonsai Ext_HighFeatures expC/sparse/bonsai/Ext_HighFeatures/0
garden Cam_Pinhole expC/sparse/garden/Cam_Pinhole/0
bonsai Cam_Pinhole expC/sparse/bonsai/Cam_Pinhole/0
garden Ext_LowFeatures expC/sparse/garden/Ext_LowFeatures/0
bonsai Ext_LowFeatures expC/sparse/bonsai/Ext_LowFeatures/0
garden
Match_Strict
Match_Loose
bonsai
bonsai
$\begin{array}{lr}\text { bonsai } & \text { Match_Loose } \\ \text { garden } & \text { Map_Robust }\end{array}$
ap_Robust
Map Robust
en/Match_Strict/0
EXCHOSe/0
expC/sparse/bonsai/Match_Strict/0
expC/sparse/bonsai/Match_Loose/0
expC/sparse/garden/Map_Robust/0
rden
bonsai
expC/sparse/bonsai/Map_Robust/0

