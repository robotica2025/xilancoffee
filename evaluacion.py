from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
import json
import os

def verificar_anotaciones(json_path):
    """Verifica y corrige las anotaciones del conjunto de prueba."""
    with open(json_path, "r") as f:
        data = json.load(f)

    for annotation in data["annotations"]:
        if annotation["category_id"] not in [1, 2, 3]:
            print(f"Corrigiendo anotación inválida: {annotation}")
            annotation["category_id"] = max(1, min(3, annotation["category_id"]))

    # Sobrescribir el archivo corregido
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    print("Anotaciones verificadas y corregidas.")

def main():
    # Ruta a los datos y el modelo
    json_path = "/Users/mr.walter/Documents/Coffee/dataset/result.json"
    images_path = "/Users/mr.walter/Documents/Coffee/dataset/images"
    output_dir = "./output"

    # Verificar y corregir anotaciones
    verificar_anotaciones(json_path)

    # Registrar conjunto de prueba
    register_coco_instances("my_dataset_test", {}, json_path, images_path)
    MetadataCatalog.get("my_dataset_test").thing_classes = ["Cultivo de Cafe", "Maleza", "Planta"]

    # Configuración
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(output_dir, "model_final.pth")  # Ruta al modelo entrenado
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25  # Reduce el umbral para capturar más detecciones
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.DEVICE = "cpu"  # Cambia a "cuda" si tienes GPU disponible
    cfg.DATALOADER.NUM_WORKERS = 0  # Deshabilita multiprocesamiento
    cfg.INPUT.MIN_SIZE_TEST = 720  # Redimensiona a 720p
    cfg.INPUT.MAX_SIZE_TEST = 1080

    # Evaluación
    evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, "my_dataset_test")

    # Predicción y resultados
    print("Iniciando evaluación...")
    predictor = DefaultPredictor(cfg)
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    print("\nResultados de la evaluación:")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
