import cv2
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class AdvancedFeatureExtractor:
    """Extractor de características avanzadas para reconocimiento facial"""

    def __init__(self):
        self.gabor_kernels = self._create_gabor_kernels()
        self.orb = cv2.ORB_create(nfeatures=100)

    def _create_gabor_kernels(self) -> List[np.ndarray]:
        """Crear banco de filtros de Gabor"""
        kernels = []

        # Parámetros de Gabor
        ksize = 21  # Tamaño del kernel
        sigma = 3.0  # Desviación estándar
        lambd = 10.0  # Longitud de onda
        gamma = 0.5  # Relación de aspecto

        # Crear kernels para diferentes orientaciones
        for theta in np.arange(0, np.pi, np.pi / 8):
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma,
                theta,
                lambd,
                gamma,
                0,
                ktype=cv2.CV_32F
            )
            kernels.append(kernel)

        return kernels

    def extract_gabor_features(self, image: np.ndarray) -> np.ndarray:
        """Extraer características usando filtros de Gabor"""
        try:
            gabor_features = []

            for kernel in self.gabor_kernels:
                # Aplicar filtro
                filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)

                # Extraer estadísticas
                gabor_features.extend([
                    np.mean(filtered),
                    np.std(filtered),
                    np.var(filtered),
                    np.percentile(filtered, 25),
                    np.percentile(filtered, 75)
                ])

            return np.array(gabor_features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error en extracción Gabor: {e}")
            return np.zeros(40)  # 8 orientaciones * 5 estadísticas

    def extract_orb_features(self, image: np.ndarray) -> np.ndarray:
        """Extraer características ORB"""
        try:
            # Detectar keypoints y descriptores
            keypoints, descriptors = self.orb.detectAndCompute(image, None)

            if descriptors is None or len(descriptors) == 0:
                return np.zeros(128)

            # Crear histograma de descriptores
            hist, _ = np.histogram(
                descriptors.flatten(),
                bins=128,
                range=(0, 256)
            )

            # Normalizar
            hist_normalized = hist.astype(np.float32) / (hist.sum() + 1e-7)

            return hist_normalized

        except Exception as e:
            logger.error(f"Error en extracción ORB: {e}")
            return np.zeros(128)

    def extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extraer características de bordes"""
        try:
            # Detectar bordes con Canny
            edges = cv2.Canny(image, 50, 150)

            # Dividir imagen en regiones
            h, w = edges.shape
            regions = [
                edges[0:h // 2, 0:w // 2],  # Superior izquierda
                edges[0:h // 2, w // 2:w],  # Superior derecha
                edges[h // 2:h, 0:w // 2],  # Inferior izquierda
                edges[h // 2:h, w // 2:w]  # Inferior derecha
            ]

            features = []
            for region in regions:
                # Densidad de bordes por región
                edge_density = np.sum(region > 0) / region.size
                features.append(edge_density)

                # Orientación dominante
                if np.any(region > 0):
                    lines = cv2.HoughLines(region, 1, np.pi / 180, 50)
                    if lines is not None:
                        angles = lines[:, 0, 1]
                        dominant_angle = np.median(angles)
                        features.append(dominant_angle)
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error en extracción de bordes: {e}")
            return np.zeros(8)

    def extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extraer características de color (para imágenes RGB)"""
        try:
            if len(image.shape) == 2:
                # Si es escala de grises, crear versión RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image

            # Convertir a diferentes espacios de color
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

            features = []

            # Histogramas por canal
            for i in range(3):
                # RGB
                hist_rgb = cv2.calcHist([image_rgb], [i], None, [32], [0, 256])
                features.extend(hist_rgb.flatten() / (hist_rgb.sum() + 1e-7))

                # HSV
                hist_hsv = cv2.calcHist([hsv], [i], None, [32], [0, 256])
                features.extend(hist_hsv.flatten() / (hist_hsv.sum() + 1e-7))

            # Momentos de color
            moments = cv2.moments(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY))
            hu_moments = cv2.HuMoments(moments).flatten()
            features.extend(hu_moments)

            return np.array(features[:256], dtype=np.float32)  # Limitar tamaño

        except Exception as e:
            logger.error(f"Error en extracción de color: {e}")
            return np.zeros(256)

    def normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """Normalización avanzada de iluminación"""
        try:
            if len(image.shape) == 2:
                # Imagen en escala de grises
                # Aplicar CLAHE
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                normalized = clahe.apply(image)

                # Ecualización adicional
                normalized = cv2.equalizeHist(normalized)

            else:
                # Imagen en color - trabajar en espacio LAB
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)

                # Aplicar CLAHE solo al canal L
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l_clahe = clahe.apply(l)

                # Recombinar canales
                lab_clahe = cv2.merge([l_clahe, a, b])
                normalized = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

            return normalized

        except Exception as e:
            logger.error(f"Error en normalización de iluminación: {e}")
            return image