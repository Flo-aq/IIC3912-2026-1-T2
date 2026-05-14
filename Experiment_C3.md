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

In [3]:

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

In [4]:
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

In [5]:

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

In [6]:

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

In [8]:

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

In [62]:

```
CONFIGS_EXEC = [
    {
        "name" : "Cam_OpenCV",
        "model" : "OPENCV",
        "ext" : "",
        "match" : "",
        "map" : "",
    },
    {
        "name" : "Ext_HighFeatures",
        "model" : "OPENCV",
        "ext" : "--SiftExtraction.max_num_features 8000",
        "match" : "",
        "map" : "",
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
    {
        "name" : "Map_Robust",
        "model" : "OPENCV",
        "ext" : "",
        "match" : "",
        "map" : "--Mapper.min_model_size 10",
    },
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

In [9]:

```
import pandas as pd
from IPython.display import display, Markdown
csv_path = f"{RESULTS_DIR}/results_expC.csv"
df = pd.read_csv(csv_path)
df_bonsai = df[df['scene'] == 'bonsai']
df_garden = df[df['scene'] == 'garden']
```


## Resumen de métricas

In [10]:

```
def resumen_escena(df_escena, nombre):
    if df_escena.empty:
        return ""
    mejor_error = df_escena.loc[df_escena['mean_reprojection_error'].idxmin()]
    peor_error = df_escena.loc[df_escena['mean_reprojection_error'].idxmax()]
    mas_puntos = df_escena.loc[df_escena['points_3d'].idxmax()]
    mejor_track = df_escena.loc[df_escena['avg_track_length'].idxmax()]
    t_min_idx = df_escena['execution_time_sec'].idxmin()
    t_max_idx = df_escena['execution_time_sec'].idxmax()
    tiempo_min_conf = df_escena.loc[t_min_idx, 'config']
    tiempo_max_conf = df_escena.loc[t_max_idx, 'config']
    tiempo_min = df_escena.loc[t_min_idx, 'execution_time_sec']
    tiempo_max = df_escena.loc[t_max_idx, 'execution_time_sec']
    # Promedios
    prom_error = df_escena['mean_reprojection_error'].mean()
    prom_puntos = df_escena['points_3d'].mean()
    prom_track = df_escena['avg_track_length'].mean()
    texto = f"""
### Escena: **{nombre}**
| Métrica | Valor | Configuración destacada |
|---------|-------||-----------------------
| Menor error de reproyección | {mejor_error['mean_reprojection_error']:.3f} px | {mejor_error['config']} |
| Mayor error de reproyección | {peor_error['mean_reprojection_error']:.3f} px | {peor_error['config']} |
| Más puntos 3D | {mas_puntos['points_3d']:,} | {mas_puntos['config']} |
| Track length más alto | {mejor_track['avg_track_length']:.2f} | {mejor_track['config']} |
| Tiempo mínimo | {tiempo_min:.0f} s | {tiempo_min_conf} |
| Tiempo máximo | {tiempo_max:.0f} s | {tiempo_max_conf} |
*Promedios (todas las configuraciones):*
- Error: {prom_error:.3f} px
- Puntos 3D: {prom_puntos:,.0f}
- Track length: {prom_track:.2f}
"""
    return texto
display(Markdown(resumen_escena(df_garden, "Garden (outdoor)")))
display(Markdown(resumen_escena(df_bonsai, "Bonsai (indoor)")))
```

Escena: Garden (outdoor)
| Métrica | Valor | Configuración destacada |
| :--- | :--- | :--- |
| Menor error de reproyección | 1.166 px | Match_Strict |
| Mayor error de reproyección | 1.326 px | Ext_LowFeatures |
| Más puntos 3D | 148,290 | Match_Loose |
| Track length más alto | 6.96 | Match_Strict |
| Tiempo mínimo | 1311 s | Ext_LowFeatures |
| Tiempo máximo | 4723 s | Match_Strict |


Promedios (todas las configuraciones):

- Error: 1.239 px
- Puntos 3D: 134,390
- Track length: 6.64

Escena: Bonsai (indoor)
| Métrica | Valor | Configuración destacada |
| :--- | :--- | :--- |
| Menor error de reproyección | 0.372 px | Match_Strict |
| Mayor error de reproyección | 1.148 px | Match_Loose |
| Más puntos 3D | 100,162 | Cam_OpenCV |
| Track length más alto | 10.09 | Match_Strict |
| Tiempo mínimo | 2048 s | Ext_LowFeatures |
| Tiempo máximo | 3938 s | Match_Loose |


Promedios (todas las configuraciones):

- Error: 0.636 px
- Puntos 3D: 71,639
- Track length: 8.33

(a)

```
In [11]:
cols = ['scene', 'config', 'points_3d', 'mean_reprojection_error', 'avg_track_length', 'execution_time_sec']
tabla_a = df[cols].copy()
tabla_a = tabla_a.sort_values(['scene', 'points_3d'], ascending=[True, False])
tabla_a.columns = ['Escena', 'Configuración', 'Puntos 3D', 'Error reproy. (px)', 'Track length prom.', 'Tiempo (s)']
display(tabla_a)
```

|  | Escena | Configuración | Puntos 3D | Error reproy. (px) | Track length prom. | Tiempo (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2 | bonsai | Cam_OpenCV | 100162 | 0.377986 | 9.484176 | 2416.904538 |
| 3 | bonsai | Ext_HighFeatures | 98402 | 0.380774 | 9.655861 | 2415.084916 |
| 7 | bonsai | Ext_LowFeatures | 94480 | 0.382625 | 9.004837 | 2048.094259 |
| 10 | bonsai | Match_Strict | 74536 | 0.372138 | 10.093136 | 2922.014849 |
| 11 | bonsai | Match_Loose | 71672 | 1.148019 | 6.874651 | 3938.032947 |
| 5 | bonsai | Cam_Pinhole | 60813 | 1.101698 | 7.554059 | 3597.550094 |
| 13 | bonsai | Map_Robust | 1410 | 0.688297 | 5.675177 | 2271.660633 |
| 9 | garden | Match_Loose | 148290 | 1.236637 | 6.585886 | 4067.003433 |
| 12 | garden | Map_Robust | 146085 | 1.235153 | 6.651997 | 4041.641545 |
| 0 | garden | Cam_OpenCV | 146084 | 1.235573 | 6.651961 | 4028.128087 |
| 4 | garden | Cam_Pinhole | 146010 | 1.235309 | 6.653318 | 3933.363019 |
| 1 | garden | Ext_HighFeatures | 142190 | 1.241969 | 6.671397 | 3868.161248 |
| 8 | garden | Match_Strict | 130468 | 1.166192 | 6.959055 | 4723.318166 |
| 6 | garden | Ext_LowFeatures | 81606 | 1.325606 | 6.281242 | 1310.753037 |

(b)

In [32]: \%pip install seaborn -q
Note: you may need to restart the kernel to use updated packages.
In [12]:

```
import seaborn as sns
markers_map = {'garden': 'o', 'bonsai': 's'}
```

In [13]:

```
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='points_3d',
    y='mean_reprojection_error',
    hue='config', # color por configuración
    style='scene', # marcador por escena
    markers=markers_map,
    s=150,
    palette='Set2'
)
plt.title('Trade-off: densidad de puntos vs precisión geométrica')
plt.xlabel('Número de puntos 3D')
plt.ylabel('Error de reproyección promedio (px)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Configuración / Escena')
plt.grid(True, linestyle='--', alpha=0.5)
```

Trade-off: densidad de puntos vs precisión geométrica
![](https://cdn.mathpix.com/cropped/30ae4453-eae9-4a0d-accf-4880142c5bd8-07.jpg?height=930&width=1278&top_left_y=246&top_left_x=273)

| Configuración / Escena |
| :--- |
| config |
| Cam_OpenCV |
| Ext_HighFeatures |
| Cam_Pinhole |
| Ext_LowFeatures |
| Match_Strict |
| Match_Loose |
| Map_Robust |
| scene |
| garden |
| bonsai |

In [14]:

```
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df,
    x='config',
    y='avg_track_length',
    hue='scene',
    palette='muted'
)
plt.title('Comparación de Robustez (Avg Track Length) por Escena', fontsize=14)
plt.xticks(rotation=45)
plt.ylabel('Longitud promedio del Track', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

Comparación de Robustez (Avg Track Length) por Escena
![](https://cdn.mathpix.com/cropped/30ae4453-eae9-4a0d-accf-4880142c5bd8-07.jpg?height=797&width=1688&top_left_y=1755&top_left_x=273)

(c)

```
In [15]:
from pycolmap import Reconstruction
from pathlib import Path
configs_focus = [
    'Cam_OpenCV',
    'Ext_HighFeatures',
    'Match_Strict'
]
palette = {
    'bonsai': '#1f77b4',
    'garden': '#ff7f0e'
}
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for i, cfg in enumerate(configs_focus):
    df_cfg = df[df['config'] == cfg]
    for _, row in df_cfg.iterrows():
        reco = pycolmap.Reconstruction(row['sparse_path'])
```

```
    lengths = [
    p.track.length()
]
sns.kdeplot(
    lengths,
    ax=axes[i],
    fill=True,
    label=row['scene'],
    color=palette[row['scene']],
    bw_adjust=1.4
)
s[i].set_title(cfg)
s[i].set_xlim(2, 15)
axes[i].set_xlabel('Track Length')
axes[i].grid(alpha=0.2)
axes[0].set_ylabel('Densidad')
axes[0].legend()
plt.suptitle('Distribución de Robustez de Tracks', fontsize=16)
plt.tight_layout()
plt.show()
```

Distribución de Robustez de Tracks
![](https://cdn.mathpix.com/cropped/30ae4453-eae9-4a0d-accf-4880142c5bd8-08.jpg?height=381&width=1697&top_left_y=924&top_left_x=267)

In [16]:

```
df_eff = df.copy()
df_eff['points_per_sec'] = (
    df_eff['points_3d'] / df_eff['execution_time_sec']
)
plt.figure(figsize=(11,6))
sns.barplot(
    data=df_eff,
    x='config',
    y='points_per_sec',
    hue='scene',
    palette={'bonsai': '#1f77b4', 'garden': '#ff7f0e'}
)
plt.title('Eficiencia del Pipeline (Puntos 3D por segundo)', fontsize=16)
plt.ylabel('Puntos 3D / segundo')
plt.xlabel('Configuración')
plt.xticks(rotation=30)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

Eficiencia del Pipeline (Puntos 3D por segundo)
![](https://cdn.mathpix.com/cropped/30ae4453-eae9-4a0d-accf-4880142c5bd8-08.jpg?height=860&width=1685&top_left_y=2080&top_left_x=276)

[^0]```
path_garden_imgs = get_image_path('garden')
path_bonsai_imgs = get_image_path('bonsai')
archivos_garden = sorted(list(path_garden_imgs.glob("*.JPG")))
archivos_bonsai = sorted(list(path_bonsai_imgs.glob("*.JPG")))
path_garden_1 = archivos_garden[0]
path_bonsai_1 = archivos_bonsai[0]
path_garden_2 = archivos_garden[1]
path_bonsai_2 = archivos_bonsai[1]
img_garden_1_name = path_garden_1.name
img_bonsai_1_name = path_bonsai_1.name
img_garden_2_name = path_garden_2.name
img_bonsai_2_name = path_bonsai_2.name
db_garden_path = DB_DIR / "garden_Cam_OpenCV.db"
db_bonsai_path = DB_DIR / "bonsai_Cam_OpenCV.db"
Imagen Garden 1: DSC07956.JPG
Imagen Garden 2: DSC07957.JPG
Imagen Bonsai 1: DSCF5565.JPG
Imagen Bonsai 2: DSCF5566.JPG
import sqlite3
import numpy as np
def get_matches_from_db(db_path, img_name1, img_name2):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 1. Obtener IDs de las imágenes
    cursor.execute("SELECT image_id FROM images WHERE name=?", (img_name1,))
    id1 = cursor.fetchone()[0]
    cursor.execute("SELECT image_id FROM images WHERE name=?", (img_name2,))
    id2 = cursor.fetchone()[0]
    # Asegurar que id1 < id2 (así los guarda COLMAP)
    pair_id = id1 * 2147483647 + id2 if id1 < id2 else id2 * 2147483647 + id1
    # 2. Obtener Keypoints
    cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (id1,))
    kps1_raw = cursor.fetchone()
    kps1 = np.frombuffer(kps1_raw[2], dtype=np.float32).reshape(kps1_raw[0], kps1_raw[1])
    cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (id2,))
    kps2_raw = cursor.fetchone()
    kps2 = np.frombuffer(kps2_raw[2], dtype=np.float32).reshape(kps2_raw[0], kps2_raw[1])
    # 3. Obtener Matches (usamos two_view_geometries porque son los matches VALIDADOS)
    cursor.execute("SELECT rows, cols, data FROM two_view_geometries WHERE pair_id=?", (pair_id,))
    matches_raw = cursor.fetchone()
    if matches_raw is None:
        return None
    matches = np.frombuffer(matches_raw[2], dtype=np.uint32).reshape(matches_raw[0], matches_raw[1])
    conn.close()
    return kps1, kps2, matches
def compare_config_matches(scene_name, config_1, config_2):
    # 1. Obtener paths de imágenes (usando tu función anterior)
    path_dir = get_image_path(scene_name)
    imgs = sorted(list(path_dir.glob("*.JPG")))
    img1_p, img2_p = imgs[0], imgs[1]
    # 2. Definir rutas de las bases de datos (ajusta según tu estructura)
    db1_path = DB_DIR / f"{scene_name}_{config_1}.db"
    db2_path = DB_DIR / f"{scene_name}_{config_2}.db"
    # 3. Extraer matches de ambas DBs
    data1 = get_matches_from_db(db1_path, img1_p.name, img2_p.name)
    data2 = get_matches_from_db(db2_path, img1_p.name, img2_p.name)
    if scene_name == 'garden':
        thickness = 2
    else:
        thickness = 1
    # 4. Graficar
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))
    for i, (data, cfg, title) in enumerate([
        (data1, config_1, "MÁXIMA DENSIDAD"),
        (data2, config_2, "MÍNIMA DENSIDAD")
    ]):
        if data:
            kps1, kps2, matches = data
            # Reutilizamos la lógica de dibujo manual sobre una imagen combinada
            img1 = cv2.imread(str(img1_p))
            img2 = cv2.imread(str(img2_p))
            h, w = img1.shape[:2]
            out = np.zeros((h, w*2, 3), dtype=np.uint8)
            out[:, :w] = img1
            out[:, w:] = img2
            # Dibujar solo los primeros n_matches
            for m in matches:
                p1 = tuple(map(int, kps1[m[0], :2]))
                p2 = (int(kps2[m[1], 0] + w), int(kps2[m[1], 1]))
                color = (0, 255, 0)
                cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)
                cv2.circle(out, p1, 2*thickness, color, -1)
                cv2.circle(out, p2, 2*thickness, color, -1)
            axes[i].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"Escena: {scene_name.upper()} | Config: {cfg} ({title})", fontsize=14)
        else:
            axes[i].text(0.5, 0.5, f"No hay matches en DB para {cfg}", ha='center')
        axes[i].axis('off')
```

```
plt.tight_layout()
plt.show()
# Uso:
# compare_config_matches('garden', 'Ext_HighFeatures', 'Ext_LowFeatures')
```

In [54]: for escena in df['scene'].unique():
\# Obtener los nombres de las configs desde los indices calculados
c_max = df.loc[idx_max[escena], 'config']
c_min = df.loc[idx_min[escena], 'config']
print(f"\n--- Generando comparativa para: \{escena.upper()\} ---")\}
compare_config_matches(escena, c_max, c_min)
--- Generando comparativa para: GARDEN ---

Escena: GARDEN | Config: Match_Loose (MÁXIMA DENSIDAD)
![](https://cdn.mathpix.com/cropped/30ae4453-eae9-4a0d-accf-4880142c5bd8-10.jpg?height=557&width=1682&top_left_y=576&top_left_x=276)

Escena: GARDEN | Config: Ext_LowFeatures (MÍNIMA DENSIDAD)
![](https://cdn.mathpix.com/cropped/30ae4453-eae9-4a0d-accf-4880142c5bd8-10.jpg?height=562&width=1678&top_left_y=1279&top_left_x=278)

Escena: BONSAI | Config: Cam_OpenCV (MÁXIMA DENSIDAD)

--- Generando comparativa para: BONSAI ---
![](https://cdn.mathpix.com/cropped/30ae4453-eae9-4a0d-accf-4880142c5bd8-10.jpg?height=570&width=1685&top_left_y=1908&top_left_x=273)

Escena: BONSAI | Config: Map_Robust (MÍNIMA DENSIDAD)
![](https://cdn.mathpix.com/cropped/30ae4453-eae9-4a0d-accf-4880142c5bd8-10.jpg?height=564&width=1680&top_left_y=2622&top_left_x=276)

\# Obtener datos de ambas DBs (usando la función que ya tenemos)
res1 = get_matches_from_db(db1_path, img_name1, img_name2)
res2 = get_matches_from_db(db2_path, img_name1, img_name2)

```
    if not res1 or not res2:
        return None
    kps1_a, kps2_a, m1 = res1
kps1_b, kps2_b, m2 = res2
# Convertir matches a coordenadas reales (x1, y1, x2, y2)
coords1 = set()
for idx1, idx2 in m1:
    coords1.add((kps1_a[idx1, 0], kps1_a[idx1, 1], kps2_a[idx2, 0], kps2_a[idx2, 1]))
coords2 = set()
for idx1, idx2 in m2:
    coords2.add((kps1_b[idx1, 0], kps1_b[idx1, 1], kps2_b[idx2, 0], kps2_b[idx2, 1]))
# Encontrar Comunes, Únicos de A y Únicos de B
# (Usamos una lógica de proximidad por si las coordenadas varían por decimales)
comunes = []
unicos_a = []
for c1 in coords1:
    found = False
    for c2 in coords2:
        # Si la distancia entre los puntos en ambas imágenes es menor al tolerance
        if np.linalg.norm(np.array(c1) - np.array(c2)) < tolerance:
            comunes.append(c1)
            found = True
            break
    if not found:
        unicos_a.append(c1)
unicos_b = [c2 for c2 in coords2 if not any(np.linalg.norm(np.array(c1) - np.array(c2)) < tolerance for c1 in comunes)]
print(f"Matches comunes: {len(comunes)} | Únicos A: {len(unicos_a)} | Únicos B: {len(unicos_b)}")
return unicos_a, unicos_b
```

In [68]:

```
def draw_comparison_matches(img1_path, img2_path, unicos_a, unicos_b, title):
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    h, w = img1.shape[:2]
    out = np.zeros((h, w*2, 3), dtype=np.uint8)
    out[:, :w] = img1
    out[:, w:] = img2
    # Configuración de dibujo (Color BGR)
    estilos = [ # Verde
        (unicos_a, (0, 0, 255), "Solo Config A"), # Rojo
        (unicos_b, (255, 0, 0), "Solo Config B") # Azul
    ]
    for lista, color, label in estilos:
        for c in lista:
            p1 = (int(c[0]), int(c[1]))
            p2 = (int(c[2] + w), int(c[3]))
            cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)
            cv2.circle(out, p1, 4, color, -1)
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    # Leyenda manual
    from matplotlib.lines import Line2D
    legend_elements = [
                Line2D([0], [0], color='r', lw=2, label='Únicos Config A'),
                    Line2D([0], [0], color='b', lw=2, label='Únicos Config B')]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.show()
```

In [69]:

```
for escena in df['scene'].unique():
    path_dir = get_image_path(escena)
    imgs = sorted(list(path_dir.glob("*.JPG")))
    img1_p, img2_p = imgs[0], imgs[1]
    c_max = df.loc[idx_max[escena], 'config']
    c_min = df.loc[idx_min[escena], 'config']
    # 2. Definir rutas de las bases de datos (ajusta según tu estructura)
    db1_path = DB_DIR / f"{escena}_{c_max}.db"
    db2_path = DB_DIR / f"{escena}_{c_min}.db"
    # 2. Definir rutas de archivos
    db_a = DB_DIR / f"{escena}_{c_max}.db"
    db_b = DB_DIR / f"{escena}_{c_min}.db"
    resultados = compare_db_matches_geom(db_a, db_b, img1_p.name, img2_p.name)
    if resultados:
        u_a, u_b = resultados
        # 4. Visualizar
        titulo = f"Estabilidad de Matches en {escena.upper()}\n{c_max} (Rojo) vs {c_min} (Azul)"
        draw_comparison_matches(img1_p, img2_p, u_a, u_b, titulo)
    else:
        print(f"Error: No se pudieron extraer matches de las DBs para la escena {escena}")
```


## Estabilidad de Matches en GARDEN Match_Loose (Rojo) vs Ext_LowFeatures (Azul)

![](https://cdn.mathpix.com/cropped/30ae4453-eae9-4a0d-accf-4880142c5bd8-12.jpg?height=557&width=1678&top_left_y=185&top_left_x=278)

Estabilidad de Matches en BONSAI Cam_OpenCV (Rojo) vs Map_Robust (Azul)

Matches comunes: 1385 | Únicos A: 1 | Únicos B: 0
![](https://cdn.mathpix.com/cropped/30ae4453-eae9-4a0d-accf-4880142c5bd8-12.jpg?height=567&width=1680&top_left_y=847&top_left_x=276)

## Exportar datos para Experimento D

```
# Guardar paths de reconstrucciones y métricas base en ./expD/
exp_d_recs = df[['scene', 'config', 'model',
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
scene config sparse_path
garden Cam_OpenCV expC/sparse/garden/Cam_OpenCV/0 garden Ext_HighFeatures expC/sparse/garden/Ext_HighFeatures/0 bonsai Cam_OpenCV expC/sparse/bonsai/Cam_OpenCV/0 bonsai Ext_HighFeatures expC/sparse/bonsai/Ext_HighFeatures/0 garden Cam_Pinhole expC/sparse/garden/Cam_Pinhole/0 bonsai Cam_Pinhole expC/sparse/bonsai/Cam_Pinhole/0 garden Ext_LowFeatures expC/sparse/garden/Ext_LowFeatures/0 bonsai Ext_LowFeatures expC/sparse/bonsai/Ext_LowFeatures/0 garden Match_Strict garden Match_Loose bonsai Match_Strict bonsai Match_Loose garden Map_Robust expC/sparse/garden/Match_Strict/0 expC/sparse/garden/Match_Loose/0 expC/sparse/bonsai/Match_Strict/0 expC/sparse/bonsai/Match_Loose/0 expC/sparse/garden/Map_Robust/0 bonsai Map_Robust expC/sparse/bonsai/Map_Robust/0


[^0]:    In [ ]: \%pip install opencv-python -q
    Note: you may need to restart the kernel to use updated packages.

    In [17]: import cv2

    In [51]: \# Encontrar las filas con el máximo y mínimo de puntos por cada escena
    idx_max = df.groupby('scene')['points_3d'].idxmax()
    idx_min = df.groupby('scene')['points_3d'].idxmin()
    config_max_garden = df.loc[idx_max['garden'], 'config']
    config_min_garden = df.loc[idx_min['garden'], 'config']
    config_max_bonsai = df.loc[idx_max['bonsai'], 'config']
    config_min_bonsai = df.loc[idx_min['bonsai'], 'config']

