import cv2
import numpy as np
import time
import json
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import serial

# Configuración del modelo
def configurar_modelo(threshold=0.2):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "./output/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)

# Inicializar la cámara
def inicializar_camara():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

# Dibujar predicciones
def dibujar_predicciones(frame, outputs, metadata):
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    scores = instances.scores if instances.has("scores") else None
    classes = instances.pred_classes if instances.has("pred_classes") else None

    COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

    for i in range(len(boxes)):
        box = boxes[i].tensor.numpy()[0]
        score = scores[i].item()
        class_id = classes[i].item()
        categoria = metadata.thing_classes[class_id]
        color = COLORS[class_id % len(COLORS)]

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        texto = f"{categoria}: {score:.2f}"
        cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# Inicializar Arduino
def inicializar_arduino(port='/dev/tty.usbmodem832401', baudrate=9600):
    try:
        arduino = serial.Serial(port=port, baudrate=baudrate, timeout=1)
        time.sleep(2)  # Esperar a que el Arduino se inicialice
        return arduino
    except Exception as e:
        print(f"Error al inicializar el Arduino: {e}")
        exit()

# Enviar comando al Arduino
def enviar_comando(arduino, comando):
    arduino.write(comando.encode())  # Enviar el comando al Arduino
    print(f"Comando enviado: {comando}")

# Main
def main():
    predictor = configurar_modelo(threshold=0.5)
    metadata = MetadataCatalog.get("my_dataset_test")
    metadata.thing_classes = ["Cultivo de Cafe", "Maleza", "Planta"]

    cap = inicializar_camara()
    arduino = inicializar_arduino()

    print("Presiona 'q' para salir.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el cuadro. Finalizando...")
            break

        frame_resized = cv2.resize(frame, (1080, 720))
        outputs = predictor(frame_resized)
        frame = dibujar_predicciones(frame, outputs, metadata)

        # Verificar si se detectó maleza
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes if instances.has("pred_classes") else []
        if 1 in pred_classes.numpy():  # Clase 1 corresponde a "Maleza"
            enviar_comando(arduino, 'H')  # Encender la MotoGuadana
            time.sleep(2)  # Mantener el pulso de  2 segundos
        else: 
            
            enviar_comando(arduino, 'L')  # Apaga MotoGuadana

        cv2.imshow("Detección de Objetos", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): #salir de la interfaz
            break

    cap.release()
    arduino.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

