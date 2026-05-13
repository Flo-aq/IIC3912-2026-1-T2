import os
import subprocess
import time
import pandas as pd
import pycolmap
import numpy as np
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS LOCALES ---
BASE_DIR = Path.cwd() / "colmap_work" 
DATASET_DIR = Path.cwd() / "data" 
LOGS_DIR = BASE_DIR / "logs_txt"

SCENES = ['bonsai'] 

def get_image_path(scene):
    path_scale = DATASET_DIR / scene / 'images_4'
    path_default = DATASET_DIR / scene / 'images'
    if path_scale.exists():
        return path_scale
    return path_default

def scene_paths(scene):
    return {
        'images': get_image_path(scene),
        'sparse': BASE_DIR / scene / 'sparse',
        'db': BASE_DIR / scene / f"colmap_{scene}.db",
        'results': BASE_DIR / "results_csv"
    }

experiment_blocks = {
    "1_CameraModels": [
        {"name": "Cam_Pinhole", "model": "PINHOLE", "ext": "", "match": "", "map": ""},
        {"name": "Cam_OpenCV", "model": "OPENCV", "ext": "", "match": "", "map": ""}
    ],
    "2_SiftExtraction": [
        {"name": "Ext_LowFeatures", "model": "OPENCV", "ext": "--SiftExtraction.max_num_features 4000", "match": "", "map": ""},
        {"name": "Ext_HighFeatures", "model": "OPENCV", "ext": "--SiftExtraction.max_num_features 10000", "match": "", "map": ""} #TODO: correr con
    ],
    "3_SiftMatching": [
        {"name": "Match_Strict", "model": "OPENCV", "ext": "", "match": "--SiftMatching.max_error 1.5", "map": ""},
        {"name": "Match_Loose", "model": "OPENCV", "ext": "", "match": "--SiftMatching.max_error 6.0", "map": ""}
    ],
    "4_Mapper": [
        {"name": "Map_Robust", "model": "OPENCV", "ext": "", "match": "", "map": "--Mapper.min_model_size 10"},
    ]
}

def execute_to_file(cmd, log_file_path, stage_name):
    print(f"  [>] Iniciando: {stage_name}...", end='', flush=True)
    start_t = time.time()
    
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n{'='*20}\nCOMANDO: {cmd}\n{'='*20}\n")
        process = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        process.wait()
    
    end_t = time.time()
    if process.returncode == 0:
        print(f" Finalizado ({end_t - start_t:.2f}s)")
    else:
        print(f" ERROR (Ver log en {log_file_path.name})")

def run_experiment(scene, conf, block_name, total_images):
    paths = scene_paths(scene)
    paths['results'].mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    if paths['db'].exists(): paths['db'].unlink()
    
    output_path = paths['sparse'] / block_name / conf['name']
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_file_path = LOGS_DIR / f"log_{scene}_{conf['name']}.txt"
    if log_file_path.exists(): log_file_path.unlink()

    extract_cmd = (f"colmap feature_extractor --database_path {paths['db']} "
                   f"--image_path {paths['images']} --ImageReader.single_camera 1 "
                   f"--ImageReader.camera_model {conf['model']} "
                   f"--SiftExtraction.use_gpu 1 {conf['ext']}")

    match_cmd = (f"colmap exhaustive_matcher --database_path {paths['db']} "
                 f"--SiftMatching.use_gpu 1 {conf['match']}")

    map_cmd = (f"colmap mapper --database_path {paths['db']} "
               f"--image_path {paths['images']} --output_path {output_path} {conf['map']}")

    print(f"\n>>> EXPERIMENTO: {scene} | {conf['name']}")
    
    start_exp = time.time()
    execute_to_file(extract_cmd, log_file_path, "Extracción")
    execute_to_file(match_cmd, log_file_path, "Matching")
    execute_to_file(map_cmd, log_file_path, "Mapping")
    end_exp = time.time()

    print(f"Tiempo Total Experimento: {end_exp - start_exp:.2f}s")

    reco_path = output_path / "0"
    if reco_path.exists():
        reco = pycolmap.Reconstruction(reco_path)
        
        # --- CÁLCULO DE MÉTRICAS AVANZADAS ---
        images_reg = reco.num_reg_images()
        points_3d = reco.num_points3D()
        num_observations = reco.compute_num_observations()
        
        metrics = {
            'scene': scene,
            'block': block_name,
            'config': conf['name'],
            'model': conf['model'],
            'images_registered': images_reg,
            'registration_ratio': images_reg / total_images if total_images > 0 else 0,
            'points_3d': points_3d,
            'mean_reprojection_error': reco.compute_mean_reprojection_error(),
            'total_observations': num_observations,
            'obs_per_point': num_observations / points_3d if points_3d > 0 else 0,
            'avg_track_length': np.mean([p.track.length() for p in reco.points3D.values()]) if points_3d > 0 else 0,
            'execution_time_sec': end_exp - start_exp
        }
        
        df = pd.DataFrame([metrics])
        df.to_csv(paths['results'] / f"res_{scene}_{conf['name']}.csv", index=False)
        return metrics
    else:
        print(f"--- [!] No se generó reconstrucción para {conf['name']} ---")
        return None
    
if __name__ == "__main__":
    # Diccionario para guardar el total de imágenes por escena
    images_count = {}

    print("=== Verificando rutas e imágenes ===")
    for scene in SCENES:
        paths = scene_paths(scene)
        if not paths['images'].exists():
            print(f"ERROR: No se encuentran imágenes para {scene} en {paths['images']}")
            exit()
        
        # Contar imágenes totales
        all_imgs = list(paths['images'].glob("*.jpg")) + list(paths['images'].glob("*.JPG"))
        images_count[scene] = len(all_imgs)
        print(f"OK: {scene} encontrada con {images_count[scene]} imágenes.")

    results_list = []
    for scene in SCENES:
        total_imgs = images_count[scene]
        for block_name, configs in experiment_blocks.items():
            for conf in configs:
                # Saltar el primero de garden (ya lo tienes)
                if scene == 'garden' and conf['name'] == 'Cam_Pinhole':
                    print(f"\n--- Saltando {scene}/{conf['name']} (Ya completado) ---")
                    continue
                
                res = run_experiment(scene, conf, block_name, total_imgs)
                if res:
                    results_list.append(res)

    if results_list:
        summary_df = pd.DataFrame(results_list)
        summary_df.to_csv(BASE_DIR / "final_summary_experiment_c.csv", index=False)
        print(f"\nResumen actualizado en: {BASE_DIR / 'final_summary_experiment_c.csv'}")