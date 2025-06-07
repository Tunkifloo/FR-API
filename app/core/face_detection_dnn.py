import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
import logging
from config import settings

logger = logging.getLogger(__name__)


class DNNFaceDetector:
    """Detector de rostros usando Deep Neural Networks"""

    def __init__(self):
        self.model_file = os.path.join(settings.MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        self.config_file = os.path.join(settings.MODELS_DIR, "deploy.prototxt")
        self.net = None
        self.confidence_threshold = 0.5
        self._load_model()

    def _load_model(self):
        """Cargar modelo DNN preentrenado"""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.config_file):
                self.net = cv2.dnn.readNetFromCaffe(self.config_file, self.model_file)
                logger.info("Modelo DNN de detección cargado exitosamente")
            else:
                logger.warning("Modelo DNN no encontrado")
                logger.info("Descarga los archivos desde:")
                logger.info(
                    "- https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")
                logger.info("- https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt")
        except Exception as e:
            logger.error(f"Error cargando modelo DNN: {e}")

    def detect_faces(self, image: np.ndarray, confidence_threshold: float = None) -> List[Tuple[int, int, int, int]]:
        """Detectar rostros usando DNN"""
        if self.net is None:
            logger.warning("Modelo DNN no disponible")
            return []

        try:
            if confidence_threshold is None:
                confidence_threshold = self.confidence_threshold

            # Obtener dimensiones
            h, w = image.shape[:2]

            # Crear blob desde imagen
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)),
                scalefactor=1.0,
                size=(300, 300),
                mean=(104.0, 177.0, 123.0)
            )

            # Configurar entrada de la red
            self.net.setInput(blob)

            # Forward pass
            detections = self.net.forward()

            faces = []

            # Procesar detecciones
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > confidence_threshold:
                    # Obtener coordenadas
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Asegurar que las coordenadas estén dentro de la imagen
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    # Convertir a formato (x, y, width, height)
                    width = endX - startX
                    height = endY - startY

                    if width > 0 and height > 0:
                        faces.append((startX, startY, width, height))

            logger.info(f"DNN detectó {len(faces)} rostros")
            return faces

        except Exception as e:
            logger.error(f"Error en detección DNN: {e}")
            return []

    def detect_faces_multiscale(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detección con múltiples umbrales de confianza"""
        all_faces = []

        # Probar con diferentes umbrales
        for threshold in [0.7, 0.5, 0.3]:
            faces = self.detect_faces(image, threshold)
            if faces:
                all_faces.extend(faces)
                break  # Si encontramos con umbral alto, no necesitamos bajar más

        # Eliminar duplicados
        return self._remove_duplicate_faces(all_faces)

    def _remove_duplicate_faces(self, faces: List[Tuple[int, int, int, int]],
                                overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """Eliminar detecciones duplicadas"""
        if len(faces) <= 1:
            return faces

        unique_faces = []

        for face in faces:
            x1, y1, w1, h1 = face
            is_duplicate = False

            for existing_face in unique_faces:
                x2, y2, w2, h2 = existing_face

                # Calcular IoU
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = x_overlap * y_overlap

                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - overlap_area

                iou = overlap_area / union_area if union_area > 0 else 0

                if iou > overlap_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_faces.append(face)

        return unique_faces