from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import json

# Función para normalizar y validar anotaciones
def normalize_annotations(dataset_name):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in dataset_dicts:
        for ann in d["annotations"]:
            if ann["category_id"] not in [1, 2, 3]:
                print(f"Anotación problemática encontrada: {ann}")
                ann["category_id"] = max(1, min(3, ann["category_id"]))

# Función para recargar y verificar el JSON
def verificar_y_corregir_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Validar y corregir `category_id` directamente en el JSON
    for annotation in data["annotations"]:
        if annotation["category_id"] not in [1, 2, 3]:
            print(f"Corrigiendo anotación inválida: {annotation}")
            annotation["category_id"] = max(1, min(3, annotation["category_id"]))

    # Sobrescribir el archivo JSON corregido
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    print("Archivo JSON verificado y corregido.")

if __name__ == "__main__":
    # Rutas al JSON y a las imágenes
    json_path = "/Users/mr.walter/Documents/Coffee/dataset/result.json"
    images_path = "/Users/mr.walter/Documents/Coffee/dataset/images"

    # Verificar y corregir el JSON antes de registrar el dataset
    verificar_y_corregir_json(json_path)

    # Registrar el dataset
    register_coco_instances("my_dataset_train", {}, json_path, images_path)

    # Validar anotaciones cargadas por Detectron2
    normalize_annotations("my_dataset_train")

    # Configuración del modelo
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2  # Mejor rendimiento si usas GPU
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2  # Reducido por limitación del dataset
    cfg.SOLVER.BASE_LR = 0.0001  # Aprendizaje más lento para un ajuste fino
    cfg.SOLVER.MAX_ITER = 2000  # Aumentado para iterar más en datos pequeños
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  # Reducción para un dataset pequeño
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Cambia según tus clases

    # Ajustes de resolución de entrada
    cfg.INPUT.MIN_SIZE_TRAIN = (720,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1080
    cfg.INPUT.MIN_SIZE_TEST = 720
    cfg.INPUT.MAX_SIZE_TEST = 1080

    # Aumentación de datos para dataset pequeño
    cfg.INPUT.RANDOM_FLIP = "horizontal"  # Flip horizontal
    cfg.INPUT.BRIGHTNESS = (0.8, 1.2)  # Brillo
    cfg.INPUT.CONTRAST = (0.8, 1.2)  # Contraste
    cfg.INPUT.SATURATION = (0.8, 1.2)  # Saturación
    cfg.INPUT.RANDOM_FLIP = "vertical"  # Añadido flip vertical para maximizar la diversidad
    cfg.MODEL.DEVICE = "cpu"  # Cambia a "cuda" si tienes GPU disponible
    # Regularización para evitar sobreajuste
    cfg.SOLVER.WEIGHT_DECAY = 0.0005  # Penalización para pesos grandes

    # Reducción del checkpoint para analizar progreso
    cfg.SOLVER.CHECKPOINT_PERIOD = 250

    # Evaluación durante el entrenamiento
    cfg.TEST.EVAL_PERIOD = 500  # Evaluación cada 500 iteraciones

   
    # Entrenamiento
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
