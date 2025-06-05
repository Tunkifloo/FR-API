import numpy as np
import cv2
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extractor avanzado de características faciales"""

    @staticmethod
    def extract_comprehensive_features(face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extraer conjunto completo de características"""
        try:
            # Redimensionar a tamaño estándar
            face_resized = cv2.resize(face_image, (128, 128))

            # 1. Características de píxeles (512 valores)
            pixel_features = face_resized.flatten().astype(np.float32) / 255.0
            pixel_subset = pixel_features[:512]

            # 2. Histograma de intensidades (256 valores)
            histogram = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
            histogram_features = histogram.flatten()

            # 3. Características de textura LBP (256 valores)
            lbp_image = FeatureExtractor._calculate_lbp(face_resized)
            lbp_histogram = cv2.calcHist([lbp_image.astype(np.uint8)], [0], None, [256], [0, 256])
            texture_features = lbp_histogram.flatten()

            # Combinar todas las características (1024 total)
            comprehensive_features = np.concatenate([
                pixel_subset,
                histogram_features,
                texture_features
            ])

            logger.info(f"Características extraídas: {len(comprehensive_features)} valores")
            return comprehensive_features

        except Exception as e:
            logger.error(f"Error extrayendo características: {e}")
            return None

    @staticmethod
    def _calculate_lbp(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Calcular Local Binary Pattern para características de textura"""
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