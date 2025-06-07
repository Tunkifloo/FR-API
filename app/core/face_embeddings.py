import cv2
import numpy as np
import os
from typing import Optional, List
import logging
from config import settings

logger = logging.getLogger(__name__)


class FaceEmbeddingExtractor:
    """Extractor de embeddings faciales usando DNN"""

    def __init__(self):
        self.model_path = os.path.join(settings.MODELS_DIR, "openface_nn4.small2.v1.t7")
        self.net = None
        self.embedding_size = 128
        self._load_model()

    def _load_model(self):
        """Cargar modelo preentrenado"""
        try:
            if os.path.exists(self.model_path):
                self.net = cv2.dnn.readNetFromTorch(self.model_path)
                logger.info("Modelo de embeddings cargado exitosamente")
            else:
                logger.warning(f"Modelo de embeddings no encontrado en {self.model_path}")
                logger.info(
                    "Descarga el modelo desde: https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7")
        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")

    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extraer embedding de 128 dimensiones"""
        if self.net is None:
            return None

        try:
            # Verificar que la imagen sea válida
            if face_image is None or face_image.size == 0:
                logger.error("Imagen facial inválida")
                return None

            # Convertir a RGB si es necesario
            if len(face_image.shape) == 2:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            else:
                face_rgb = face_image

            # Preparar imagen
            face_resized = cv2.resize(face_rgb, (96, 96))

            # Crear blob
            blob = cv2.dnn.blobFromImage(
                face_resized,
                scalefactor=1.0 / 255.0,
                size=(96, 96),
                mean=(0, 0, 0),
                swapRB=True,
                crop=False
            )

            # Forward pass
            self.net.setInput(blob)
            embedding = self.net.forward()

            # Normalizar embedding
            embedding_normalized = embedding.flatten()
            embedding_normalized = embedding_normalized / np.linalg.norm(embedding_normalized)

            return embedding_normalized

        except Exception as e:
            logger.error(f"Error extrayendo embedding: {e}")
            return None

    def extract_batch_embeddings(self, face_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Extraer embeddings de múltiples imágenes"""
        embeddings = []
        for face_image in face_images:
            embedding = self.extract_embedding(face_image)
            embeddings.append(embedding)
        return embeddings

    def calculate_embedding_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calcular distancia entre embeddings"""
        try:
            # Distancia euclidiana
            distance = np.linalg.norm(embedding1 - embedding2)

            # Convertir a similitud (0-1)
            # Típicamente las distancias están entre 0-2
            similarity = max(0, 1 - (distance / 2.0))

            return similarity
        except Exception as e:
            logger.error(f"Error calculando distancia: {e}")
            return 0.0