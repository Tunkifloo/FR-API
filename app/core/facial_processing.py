import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class FacialProcessor:
    """Procesador de imágenes faciales"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def process_image(self, image_path: str) -> Optional[np.ndarray]:
        """Procesar imagen y detectar rostros"""
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"No se pudo cargar la imagen: {image_path}")
                return None

            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Eliminar ruido
            gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
            cleaned = cv2.bilateralFilter(gaussian, 9, 75, 75)

            return cleaned

        except Exception as e:
            logger.error(f"Error procesando imagen: {e}")
            return None

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detectar rostros en imagen"""
        try:
            faces = self.face_cascade.detectMultiScale(
                image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces.tolist() if len(faces) > 0 else []
        except Exception as e:
            logger.error(f"Error detectando rostros: {e}")
            return []

    def extract_face_features(self, image: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extraer características faciales"""
        try:
            x, y, w, h = face_coords
            face_roi = image[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (128, 128))

            # Extraer características
            pixel_features = face_resized.flatten().astype(np.float32) / 255.0
            histogram = cv2.calcHist([face_resized], [0], None, [256], [0, 256]).flatten()

            # LBP para textura
            lbp_image = self._calculate_lbp(face_resized)
            lbp_histogram = cv2.calcHist([lbp_image.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()

            # Combinar características
            features = np.concatenate([
                pixel_features[:512],
                histogram,
                lbp_histogram
            ])

            return features

        except Exception as e:
            logger.error(f"Error extrayendo características: {e}")
            return None

    def _calculate_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Calcular Local Binary Pattern"""
        lbp = np.zeros_like(image)
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                lbp_value = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x_coord = int(i + radius * np.cos(angle))
                    y_coord = int(j + radius * np.sin(angle))
                    if image[x_coord, y_coord] >= center:
                        lbp_value += 2 ** k
                lbp[i, j] = lbp_value
        return lbp

    def compare_features(self, features1: np.ndarray, features2: np.ndarray, threshold: float = 0.75) -> dict:
        """Comparar características faciales"""
        try:
            # Calcular similitud por correlación
            correlation = np.corrcoef(features1, features2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0

            # Calcular similitud por distancia
            distance = np.linalg.norm(features1 - features2)
            distance_similarity = max(0, 1 - (distance / np.sqrt(len(features1))))

            # Similitud promedio
            similarity = (abs(correlation) + distance_similarity) / 2

            return {
                "similarity": float(similarity),
                "correlation": float(correlation),
                "distance": float(distance),
                "distance_similarity": float(distance_similarity),
                "is_match": similarity >= threshold,
                "threshold": threshold
            }

        except Exception as e:
            logger.error(f"Error comparando características: {e}")
            return {
                "similarity": 0.0,
                "correlation": 0.0,
                "distance": float('inf'),
                "distance_similarity": 0.0,
                "is_match": False,
                "threshold": threshold
            }