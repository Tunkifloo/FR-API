import numpy as np
import cv2
from typing import Optional, Tuple
import logging
from config import settings
from app.core.advanced_features import AdvancedFeatureExtractor
from app.core.face_embeddings import FaceEmbeddingExtractor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extractor avanzado y optimizado de características faciales"""

    def __init__(self):
        """Inicializar extractor con configuraciones del sistema"""
        self.method = getattr(settings, 'FEATURE_METHOD', 'enhanced')
        self.target_size = getattr(settings, 'TARGET_FEATURE_SIZE', 512)
        self.face_size = getattr(settings, 'FACE_SIZE', (128, 128))
        self.normalize = getattr(settings, 'NORMALIZE_FEATURES', True)

        # Inicializar extractores avanzados
        self.advanced_extractor = AdvancedFeatureExtractor()
        self.embedding_extractor = None
        if settings.USE_FACE_EMBEDDINGS:
            self.embedding_extractor = FaceEmbeddingExtractor()

    def extract_comprehensive_features(self, face_image: np.ndarray, method: Optional[str] = None) -> Optional[
        np.ndarray]:
        """Extraer conjunto completo de características con método especificado"""
        try:
            # Usar método especificado o el configurado por defecto
            extraction_method = method or self.method

            if extraction_method == "enhanced":
                return self._extract_enhanced_features(face_image)
            elif extraction_method == "traditional":
                return self._extract_traditional_features(face_image)
            else:
                logger.warning(f"Método desconocido {extraction_method}, usando enhanced")
                return self._extract_enhanced_features(face_image)

        except Exception as e:
            logger.error(f"Error extrayendo características: {e}")
            return None

    def _extract_gabor_features(self, image: np.ndarray) -> np.ndarray:
        """Extraer características usando filtros de Gabor"""
        try:
            gabor_features = []

            # Parámetros de Gabor
            num_filters = 8
            ksize = 21
            sigma = 3.0
            lambd = 10.0
            gamma = 0.5

            for theta in np.arange(0, np.pi, np.pi / num_filters):
                kern = cv2.getGaborKernel((ksize, ksize), sigma, theta,
                                          lambd, gamma, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(image, cv2.CV_8UC3, kern)

                # Estadísticas del filtro
                gabor_features.extend([
                    np.mean(filtered),
                    np.std(filtered),
                    np.var(filtered)
                ])

            return np.array(gabor_features)
        except Exception as e:
            logger.warning(f"Error en Gabor: {e}")
            return np.zeros(24)

    def _extract_orb_features(self, image: np.ndarray) -> np.ndarray:
        """Extraer características ORB (Oriented FAST and Rotated BRIEF)"""
        try:
            orb = cv2.ORB_create(nfeatures=100)
            keypoints, descriptors = orb.detectAndCompute(image, None)

            if descriptors is None:
                return np.zeros(128)

            # Crear histograma de descriptores
            hist, _ = np.histogram(descriptors.flatten(), bins=128, range=(0, 256))
            return hist.astype(np.float32) / (hist.sum() + 1e-7)

        except Exception as e:
            logger.warning(f"Error en ORB: {e}")
            return np.zeros(128)

    def _extract_enhanced_features(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extracción mejorada de características con métodos avanzados"""
        try:
            # Normalizar iluminación primero
            face_normalized = self.advanced_extractor.normalize_illumination(face_image)

            # Redimensionar a tamaño estándar
            face_resized = cv2.resize(face_normalized, self.face_size)
            features_list = []

            # 1. Características de píxeles normalizadas
            pixel_features = face_resized.flatten().astype(np.float32) / 255.0
            pixel_subset = pixel_features[::4][:200]
            features_list.append(pixel_subset)

            # 2. Histograma de intensidades
            hist = cv2.calcHist([face_resized], [0], None, [64], [0, 256])
            hist_normalized = hist.flatten() / (hist.sum() + 1e-7)
            features_list.append(hist_normalized)

            # 3. LBP multi-escala
            lbp_features = self._extract_multi_scale_lbp(face_resized)
            features_list.append(lbp_features)

            # 4. HOG simplificado
            hog_features = self._extract_simple_hog(face_resized)
            features_list.append(hog_features)

            # 5. Simetría facial
            symmetry_features = self._extract_symmetry_features(face_resized)
            features_list.append(symmetry_features)

            # 6. Características Gabor
            if settings.USE_GABOR_FEATURES:
                gabor_features = self.advanced_extractor.extract_gabor_features(face_resized)
                features_list.append(gabor_features)

            # 7. Características ORB
            if settings.USE_ORB_FEATURES:
                orb_features = self.advanced_extractor.extract_orb_features(face_resized)
                features_list.append(orb_features)

            # 8. Características de bordes
            if settings.USE_EDGE_FEATURES:
                edge_features = self.advanced_extractor.extract_edge_features(face_resized)
                features_list.append(edge_features)

            # 9. Características de color
            if settings.USE_COLOR_FEATURES and len(face_image.shape) == 3:
                color_features = self.advanced_extractor.extract_color_features(face_image)
                features_list.append(color_features)

            # 10. Face Embeddings
            if self.embedding_extractor:
                embedding = self.embedding_extractor.extract_embedding(face_image)
                if embedding is not None:
                    features_list.append(embedding)
                else:
                    features_list.append(np.zeros(128))

            # Concatenar todas las características
            all_features = np.concatenate(features_list)

            # Normalización
            if self.normalize:
                mean = np.mean(all_features)
                std = np.std(all_features)
                if std > 0:
                    all_features = (all_features - mean) / std
                    all_features = np.clip(all_features, -3, 3)

            logger.info(f"Características avanzadas extraídas: {len(all_features)} valores")
            return all_features

        except Exception as e:
            logger.error(f"Error en extracción avanzada: {e}")
            return None

    def _extract_traditional_features(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Método tradicional mejorado (compatible hacia atrás)"""
        try:
            face_resized = cv2.resize(face_image, self.face_size)

            # 1. Características de píxeles con submuestreo
            pixel_features = face_resized.flatten().astype(np.float32) / 255.0
            pixel_subset = pixel_features[::3][:300]  # Submuestreo más conservador

            # 2. Histograma con menos bins para reducir ruido
            histogram = cv2.calcHist([face_resized], [0], None, [64], [0, 256])
            histogram_normalized = histogram.flatten() / (histogram.sum() + 1e-7)

            # 3. LBP básico
            lbp_image = self._calculate_lbp(face_resized)
            lbp_histogram = cv2.calcHist([lbp_image.astype(np.uint8)], [0], None, [64], [0, 256])
            texture_normalized = lbp_histogram.flatten() / (lbp_histogram.sum() + 1e-7)

            # Combinar características (total ≈ 428 valores)
            comprehensive_features = np.concatenate([
                pixel_subset,
                histogram_normalized,
                texture_normalized
            ])

            logger.info(f"Características tradicionales extraídas: {len(comprehensive_features)} valores")
            return comprehensive_features

        except Exception as e:
            logger.error(f"Error en extracción tradicional: {e}")
            return None

    def _extract_multi_scale_lbp(self, image: np.ndarray) -> np.ndarray:
        """LBP multi-escala para mejor caracterización de textura"""
        lbp_features = []

        # LBP con diferentes radios y puntos
        configs = [(1, 8), (2, 12), (3, 16)]

        for radius, n_points in configs:
            try:
                lbp = self._calculate_lbp(image, radius, n_points)
                # Usar menos bins para reducir dimensionalidad
                hist = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [16], [0, 256])
                lbp_features.extend(hist.flatten() / (hist.sum() + 1e-7))
            except Exception as e:
                logger.warning(f"Error en LBP con radio {radius}: {e}")
                # Rellenar con ceros si falla
                lbp_features.extend([0.0] * 16)

        return np.array(lbp_features)

    def _calculate_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Calcular Local Binary Pattern optimizado"""
        lbp = np.zeros_like(image, dtype=np.uint8)

        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                lbp_value = 0

                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x_coord = int(i + radius * np.cos(angle))
                    y_coord = int(j + radius * np.sin(angle))

                    # Verificar límites
                    if (0 <= x_coord < image.shape[0] and 0 <= y_coord < image.shape[1]):
                        if image[x_coord, y_coord] >= center:
                            lbp_value += 2 ** k

                lbp[i, j] = lbp_value

        return lbp

    def _extract_simple_hog(self, image: np.ndarray) -> np.ndarray:
        """HOG simplificado para capturar gradientes direccionales"""
        try:
            # Calcular gradientes
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

            # Magnitud y orientación
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            orientation = np.arctan2(grad_y, grad_x)

            # Dividir en celdas más grandes para reducir dimensionalidad
            cell_size = 16  # Celdas más grandes
            n_bins = 6  # Menos bins
            hog_features = []

            for i in range(0, image.shape[0] - cell_size, cell_size):
                for j in range(0, image.shape[1] - cell_size, cell_size):
                    cell_mag = magnitude[i:i + cell_size, j:j + cell_size]
                    cell_ori = orientation[i:i + cell_size, j:j + cell_size]

                    # Histograma de orientaciones ponderado por magnitud
                    hist, _ = np.histogram(cell_ori, bins=n_bins, range=(-np.pi, np.pi), weights=cell_mag)
                    hist_norm = hist / (np.sum(hist) + 1e-7)
                    hog_features.extend(hist_norm)

            return np.array(hog_features)

        except Exception as e:
            logger.warning(f"Error en HOG: {e}")
            return np.zeros(24)  # Retornar vector por defecto

    def _extract_symmetry_features(self, image: np.ndarray) -> np.ndarray:
        """Características de simetría facial optimizadas"""
        try:
            height, width = image.shape

            # Dividir rostro por la mitad
            left_half = image[:, :width // 2]
            right_half = image[:, width // 2:]
            right_half_flipped = cv2.flip(right_half, 1)

            # Redimensionar para asegurar mismo tamaño
            target_width = min(left_half.shape[1], right_half_flipped.shape[1])
            if target_width > 0:
                left_resized = cv2.resize(left_half, (target_width, height))
                right_resized = cv2.resize(right_half_flipped, (target_width, height))

                # Calcular diferencias
                symmetry_diff = np.abs(left_resized.astype(np.float32) - right_resized.astype(np.float32))

                # Estadísticas de simetría más robustas
                symmetry_features = [
                    np.mean(symmetry_diff),
                    np.std(symmetry_diff),
                    np.median(symmetry_diff),
                    np.percentile(symmetry_diff, 25),
                    np.percentile(symmetry_diff, 75),
                    np.max(symmetry_diff),
                    np.min(symmetry_diff)
                ]
            else:
                # Valores por defecto si hay error
                symmetry_features = [0.0] * 7

            return np.array(symmetry_features)

        except Exception as e:
            logger.warning(f"Error en simetría: {e}")
            return np.zeros(7)

    def get_feature_info(self, features: np.ndarray) -> dict:
        """Obtener información sobre las características extraídas"""
        if features is None:
            return {"error": "No features provided"}

        return {
            "total_features": len(features),
            "mean": float(np.mean(features)),
            "std": float(np.std(features)),
            "min": float(np.min(features)),
            "max": float(np.max(features)),
            "method": self.method,
            "normalized": self.normalize,
            "feature_range": [float(np.percentile(features, 25)), float(np.percentile(features, 75))],
            "zero_percentage": float(np.sum(features == 0) / len(features) * 100)
        }

    def validate_features(self, features: np.ndarray) -> Tuple[bool, str]:
        """Validar características extraídas"""
        if features is None:
            return False, "Features es None"

        if len(features) == 0:
            return False, "Vector de características vacío"

        if len(features) < 50:
            return False, "Vector de características muy pequeño"

        if np.all(features == 0):
            return False, "Todas las características son cero"

        if np.any(np.isnan(features)):
            return False, "Características contienen NaN"

        if np.any(np.isinf(features)):
            return False, "Características contienen infinito"

        # Verificar si hay suficiente variación
        if np.std(features) < 1e-6:
            return False, "Características sin variación significativa"

        return True, "Características válidas"

    def compare_feature_vectors(self, features1: np.ndarray, features2: np.ndarray) -> dict:
        """Comparar dos vectores de características"""
        try:
            if len(features1) != len(features2):
                return {
                    "compatible": False,
                    "reason": f"Dimensiones diferentes: {len(features1)} vs {len(features2)}"
                }

            # Calcular estadísticas de comparación
            correlation = np.corrcoef(features1, features2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0

            cosine_sim = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
            if np.isnan(cosine_sim):
                cosine_sim = 0.0

            euclidean_dist = np.linalg.norm(features1 - features2)

            return {
                "compatible": True,
                "dimensions": len(features1),
                "correlation": float(correlation),
                "cosine_similarity": float(cosine_sim),
                "euclidean_distance": float(euclidean_dist),
                "normalized_distance": float(euclidean_dist / np.sqrt(len(features1)))
            }

        except Exception as e:
            return {
                "compatible": False,
                "reason": f"Error en comparación: {str(e)}"
            }

    @staticmethod
    def extract_features_from_image_path(image_path: str, method: str = None) -> Optional[np.ndarray]:
        """Extraer características directamente desde ruta de imagen"""
        try:
            # Cargar imagen
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.error(f"No se pudo cargar imagen: {image_path}")
                return None

            # Crear extractor y procesar
            extractor = FeatureExtractor()
            return extractor.extract_comprehensive_features(image, method)

        except Exception as e:
            logger.error(f"Error extrayendo desde {image_path}: {e}")
            return None

    @staticmethod
    def batch_extract_features(images: list, method: str = None) -> list:
        """Extraer características de múltiples imágenes"""
        results = []
        extractor = FeatureExtractor()

        for i, image in enumerate(images):
            try:
                if isinstance(image, str):
                    # Es una ruta de archivo
                    features = FeatureExtractor.extract_features_from_image_path(image, method)
                elif isinstance(image, np.ndarray):
                    # Es una imagen en memoria
                    features = extractor.extract_comprehensive_features(image, method)
                else:
                    logger.warning(f"Tipo de imagen no soportado en índice {i}")
                    features = None

                results.append({
                    "index": i,
                    "features": features,
                    "success": features is not None,
                    "feature_count": len(features) if features is not None else 0
                })

            except Exception as e:
                logger.error(f"Error procesando imagen {i}: {e}")
                results.append({
                    "index": i,
                    "features": None,
                    "success": False,
                    "error": str(e)
                })

        return results

    def get_extraction_config(self) -> dict:
        """Obtener configuración actual del extractor"""
        return {
            "method": self.method,
            "target_size": self.target_size,
            "face_size": self.face_size,
            "normalize": self.normalize,
            "version": "2.0",
            "supported_methods": ["enhanced", "traditional"]
        }


# Funciones de utilidad para compatibilidad hacia atrás
def extract_comprehensive_features(face_image: np.ndarray) -> Optional[np.ndarray]:
    """Función de compatibilidad con código existente"""
    extractor = FeatureExtractor()
    return extractor.extract_comprehensive_features(face_image)


def extract_features_enhanced(face_image: np.ndarray) -> Optional[np.ndarray]:
    """Extracción con método mejorado específicamente"""
    extractor = FeatureExtractor()
    return extractor.extract_comprehensive_features(face_image, method="enhanced")


def extract_features_traditional(face_image: np.ndarray) -> Optional[np.ndarray]:
    """Extracción con método tradicional específicamente"""
    extractor = FeatureExtractor()
    return extractor.extract_comprehensive_features(face_image, method="traditional")