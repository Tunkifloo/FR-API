import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging
import os
from config import settings
from app.core.face_detection_dnn import DNNFaceDetector
from app.core.face_embeddings import FaceEmbeddingExtractor
from app.core.advanced_features import AdvancedFeatureExtractor

# Solo importar sklearn si está disponible
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn no disponible, usando comparación básica")

logger = logging.getLogger(__name__)


class FacialProcessor:
    """Procesador de imágenes faciales con capacidades mejoradas"""

    def __init__(self):
        # Cascadas para detección (siempre disponibles con OpenCV)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )

        # Inicializar detectores adicionales
        self.dnn_detector = None
        if settings.USE_DNN_DETECTION:
            self.dnn_detector = DNNFaceDetector()

        # Inicializar extractores avanzados
        self.embedding_extractor = None
        if settings.USE_FACE_EMBEDDINGS:
            self.embedding_extractor = FaceEmbeddingExtractor()

        self.advanced_extractor = AdvancedFeatureExtractor()

        # dlib no se usa más para evitar problemas de instalación
        self.use_dlib = False

        # Configuración del procesamiento
        self.enhanced_mode = getattr(settings, 'USE_ENHANCED_PROCESSING', True)

        logger.info(f"FacialProcessor inicializado - Modo mejorado: {self.enhanced_mode}")
        logger.info(f"DNN Detection: {settings.USE_DNN_DETECTION}")
        logger.info(f"Face Embeddings: {settings.USE_FACE_EMBEDDINGS}")

    def process_image(self, image_path: str) -> Optional[np.ndarray]:
        """Procesar imagen básico (compatibilidad hacia atrás)"""
        if self.enhanced_mode:
            return self.preprocess_image(image_path)
        else:
            return self._process_image_basic(image_path)

    def detect_faces_hybrid(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detección híbrida usando múltiples métodos"""
        all_faces = []

        # 1. Intentar con DNN primero (más preciso)
        if self.dnn_detector:
            dnn_faces = self.dnn_detector.detect_faces_multiscale(image)
            if dnn_faces:
                logger.info(f"DNN detectó {len(dnn_faces)} rostros")
                return dnn_faces  # Si DNN encuentra rostros, confiar en él

        # 2. Fallback a detección robusta con cascadas
        cascade_faces = self.detect_faces_robust(image)

        return cascade_faces

    def detect_faces_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detección de rostros usando DNN (más precisa)"""
        try:
            # Cargar modelo DNN preentrenado de OpenCV
            modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
            configFile = "models/deploy.prototxt"

            if not os.path.exists(modelFile) or not os.path.exists(configFile):
                logger.warning("Modelo DNN no encontrado, usando cascadas")
                return self.detect_faces_robust(image)

            net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

            # Preparar imagen
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                         (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            faces = []
            h, w = image.shape[:2]

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Umbral de confianza
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x2, y2) = box.astype("int")
                    faces.append((x, y, x2 - x, y2 - y))

            return faces
        except Exception as e:
            logger.error(f"Error en detección DNN: {e}")
            return self.detect_faces_robust(image)

    def _calculate_mahalanobis_distance(self, features1: np.ndarray,
                                        features2: np.ndarray,
                                        covariance_matrix: Optional[np.ndarray] = None) -> float:
        """Calcular distancia de Mahalanobis (considera correlaciones)"""
        try:
            if covariance_matrix is None:
                # Usar matriz identidad si no hay covarianza
                covariance_matrix = np.eye(len(features1))

            diff = features1 - features2
            inv_cov = np.linalg.inv(covariance_matrix)
            distance = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))

            # Normalizar a similitud
            similarity = 1 / (1 + distance)
            return similarity

        except Exception as e:
            logger.warning(f"Error en Mahalanobis: {e}")
            return 0.0

    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocesamiento mejorado de imagen"""
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"No se pudo cargar la imagen: {image_path}")
                return None

            logger.info(f"Imagen cargada - Tamaño original: {image.shape}")

            # Redimensionar si es muy grande (optimización)
            height, width = image.shape[:2]
            if width > 1024 or height > 1024:
                scale = min(1024 / width, 1024 / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
                logger.info(f"Imagen redimensionada a: {new_width}x{new_height}")

            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Mejorar contraste y brillo
            gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)

            # Filtrado adaptativo para reducir ruido
            gray = cv2.bilateralFilter(gray, 9, 75, 75)

            # Ecualización de histograma adaptativa
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            logger.info(f"Preprocesamiento completado - Imagen final: {gray.shape}")
            return gray

        except Exception as e:
            logger.error(f"Error preprocesando imagen: {e}")
            return None

    def _process_image_basic(self, image_path: str) -> Optional[np.ndarray]:
        """Procesamiento básico original"""
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
        """Detectar rostros básico (compatibilidad hacia atrás)"""
        if self.enhanced_mode and getattr(settings, 'USE_MULTIPLE_DETECTORS', True):
            return self.detect_faces_robust(image)
        else:
            return self._detect_faces_basic(image)

    def detect_faces_robust(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detección robusta de rostros con múltiples métodos y parámetros más sensibles"""
        faces = []

        try:
            logger.info(f"Iniciando detección robusta en imagen de tamaño: {image.shape}")

            # Método 1: Cascade frontal con parámetros MÁS SENSIBLES
            logger.info("Probando detección frontal estricta...")
            faces_frontal1 = self.face_cascade.detectMultiScale(
                image,
                scaleFactor=1.05,  # Más granular
                minNeighbors=3,  # MÁS PERMISIVO (era 6)
                minSize=(30, 30),  # TAMAÑO MÍNIMO MENOR (era 50,50)
                maxSize=(800, 800),  # TAMAÑO MÁXIMO MAYOR
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces_frontal1) > 0:
                logger.info(f"Detección frontal estricta encontró {len(faces_frontal1)} rostros")
                faces.extend(faces_frontal1.tolist())

            # Método 2: Cascade frontal con parámetros MUY PERMISIVOS
            if len(faces) == 0:
                logger.info("Probando detección frontal permisiva...")
                faces_frontal2 = self.face_cascade.detectMultiScale(
                    image,
                    scaleFactor=1.1,  # Menos granular pero más rápido
                    minNeighbors=2,  # MUY PERMISIVO
                    minSize=(20, 20),  # TAMAÑO MÍNIMO MUY PEQUEÑO
                    maxSize=(),  # Sin límite de tamaño máximo
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces_frontal2) > 0:
                    logger.info(f"Detección frontal permisiva encontró {len(faces_frontal2)} rostros")
                    faces.extend(faces_frontal2.tolist())

            # Método 3: Cascade de perfil si no encontramos nada
            if len(faces) == 0:
                logger.info("Probando detección de perfil...")
                faces_profile = self.profile_cascade.detectMultiScale(
                    image,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(20, 20)
                )
                if len(faces_profile) > 0:
                    logger.info(f"Detección de perfil encontró {len(faces_profile)} rostros")
                    faces.extend(faces_profile.tolist())

            # Método 4: Intentar con diferentes escalas si seguimos sin encontrar
            if len(faces) == 0:
                logger.info("Probando con escalas múltiples...")
                for scale_factor in [1.3, 1.15, 1.07]:
                    faces_scale = self.face_cascade.detectMultiScale(
                        image,
                        scaleFactor=scale_factor,
                        minNeighbors=1,  # SÚPER PERMISIVO
                        minSize=(15, 15),  # SÚPER PEQUEÑO
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    if len(faces_scale) > 0:
                        logger.info(f"Escala {scale_factor} encontró {len(faces_scale)} rostros")
                        faces.extend(faces_scale.tolist())
                        break

            # Eliminar duplicados si encontramos rostros
            if len(faces) > 0:
                faces = self._remove_duplicate_faces(faces)
                # Ordenar por tamaño (el más grande primero)
                faces.sort(key=lambda f: f[2] * f[3], reverse=True)
                logger.info(f"Total de rostros únicos detectados: {len(faces)}")
            else:
                logger.warning("NO SE DETECTARON ROSTROS con ningún método")

            return faces

        except Exception as e:
            logger.error(f"Error detectando rostros: {e}")
            return []

    def _detect_faces_basic(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detección básica con parámetros más permisivos"""
        try:
            logger.info("Usando detección básica más permisiva...")
            faces = self.face_cascade.detectMultiScale(
                image,
                scaleFactor=1.1,
                minNeighbors=3,  # Más permisivo (era 5)
                minSize=(20, 20)  # Más pequeño (era 30,30)
            )
            logger.info(f"Detección básica encontró {len(faces)} rostros")
            return faces.tolist() if len(faces) > 0 else []
        except Exception as e:
            logger.error(f"Error detectando rostros básicos: {e}")
            return []

    def _remove_duplicate_faces(self, faces: List[List[int]], overlap_threshold: float = 0.3) -> List[List[int]]:
        """Eliminar rostros duplicados basado en superposición"""
        if len(faces) <= 1:
            return faces

        unique_faces = []

        for face in faces:
            is_duplicate = False
            x1, y1, w1, h1 = face

            for existing_face in unique_faces:
                x2, y2, w2, h2 = existing_face

                # Calcular intersección
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = x_overlap * y_overlap

                # Calcular áreas
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - overlap_area

                # Calcular IoU (Intersection over Union)
                iou = overlap_area / union_area if union_area > 0 else 0

                if iou > overlap_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_faces.append(face)

        return unique_faces

    def extract_face_features(self, image: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extraer características faciales (compatibilidad hacia atrás)"""
        if self.enhanced_mode:
            return self.extract_enhanced_features(image, face_coords)
        else:
            return self._extract_face_features_basic(image, face_coords)

    def extract_enhanced_features(self, image: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Optional[
        np.ndarray]:
        """Extracción mejorada de características faciales con métodos avanzados"""
        try:
            x, y, w, h = face_coords
            logger.info(f"Extrayendo características de rostro en: x={x}, y={y}, w={w}, h={h}")

            # Expandir ROI ligeramente para mejor contexto
            expand_ratio = 0.1
            x_exp = max(0, int(x - w * expand_ratio))
            y_exp = max(0, int(y - h * expand_ratio))
            w_exp = min(image.shape[1] - x_exp, int(w * (1 + 2 * expand_ratio)))
            h_exp = min(image.shape[0] - y_exp, int(h * (1 + 2 * expand_ratio)))

            face_roi = image[y_exp:y_exp + h_exp, x_exp:x_exp + w_exp]

            # Normalizar iluminación
            face_roi = self.advanced_extractor.normalize_illumination(face_roi)

            # Redimensionar a múltiples tamaños para mejor robustez
            face_128 = cv2.resize(face_roi, (128, 128))
            face_64 = cv2.resize(face_roi, (64, 64))

            features_list = []

            # 1. Características de píxeles normalizadas (256 valores)
            pixel_features = face_128.flatten().astype(np.float32) / 255.0
            features_list.append(pixel_features[::2][:256])  # Submuestreo

            # 2. Histograma de intensidades (64 bins)
            hist = cv2.calcHist([face_128], [0], None, [64], [0, 256])
            hist_normalized = hist.flatten() / (hist.sum() + 1e-7)
            features_list.append(hist_normalized)

            # 3. LBP mejorado con múltiples radios
            lbp_features = self._extract_multi_scale_lbp(face_128)
            features_list.append(lbp_features)

            # 4. Gradientes direccionales (HOG simplificado)
            hog_features = self._extract_simple_hog(face_64)
            features_list.append(hog_features)

            # 5. Características de simetría facial
            symmetry_features = self._extract_symmetry_features(face_128)
            features_list.append(symmetry_features)

            # 6. Características Gabor (si está habilitado)
            if settings.USE_GABOR_FEATURES:
                gabor_features = self.advanced_extractor.extract_gabor_features(face_128)
                features_list.append(gabor_features)

            # 7. Características ORB (si está habilitado)
            if settings.USE_ORB_FEATURES:
                orb_features = self.advanced_extractor.extract_orb_features(face_128)
                features_list.append(orb_features)

            # 8. Características de bordes (si está habilitado)
            if settings.USE_EDGE_FEATURES:
                edge_features = self.advanced_extractor.extract_edge_features(face_128)
                features_list.append(edge_features)

            # 9. Face Embeddings (si está habilitado)
            if self.embedding_extractor:
                embedding = self.embedding_extractor.extract_embedding(face_roi)
                if embedding is not None:
                    features_list.append(embedding)
                else:
                    # Si falla, agregar vector de ceros
                    features_list.append(np.zeros(128))

            # Concatenar todas las características
            all_features = np.concatenate(features_list)

            # Normalización
            if settings.NORMALIZE_FEATURES:
                mean = np.mean(all_features)
                std = np.std(all_features)
                if std > 0:
                    all_features = (all_features - mean) / std
                    all_features = np.clip(all_features, -3, 3)

            logger.info(f"Características extraídas: {len(all_features)} valores")
            return all_features

        except Exception as e:
            logger.error(f"Error extrayendo características mejoradas: {e}")
            return None

    def _extract_face_features_basic(self, image: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Optional[
        np.ndarray]:
        """Extracción básica original"""
        try:
            x, y, w, h = face_coords
            face_roi = image[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (128, 128))

            # Extraer características básicas
            pixel_features = face_resized.flatten().astype(np.float32) / 255.0
            histogram = cv2.calcHist([face_resized], [0], None, [256], [0, 256]).flatten()

            # LBP básico para textura
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
            logger.error(f"Error extrayendo características básicas: {e}")
            return None

    def _extract_multi_scale_lbp(self, image: np.ndarray) -> np.ndarray:
        """LBP multi-escala para mejor caracterización de textura"""
        lbp_features = []

        # LBP con diferentes radios
        for radius in [1, 2, 3]:
            n_points = 8 * radius
            lbp = self._calculate_lbp(image, radius, n_points)
            hist = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [32], [0, 256])
            lbp_features.extend(hist.flatten() / (hist.sum() + 1e-7))

        return np.array(lbp_features)

    def _calculate_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """LBP optimizado"""
        lbp = np.zeros_like(image)
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                lbp_value = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x_coord = int(i + radius * np.cos(angle))
                    y_coord = int(j + radius * np.sin(angle))
                    if 0 <= x_coord < image.shape[0] and 0 <= y_coord < image.shape[1]:
                        if image[x_coord, y_coord] >= center:
                            lbp_value += 2 ** k
                lbp[i, j] = lbp_value
        return lbp

    def _extract_simple_hog(self, image: np.ndarray) -> np.ndarray:
        """HOG simplificado para capturar gradientes"""
        try:
            # Calcular gradientes
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

            # Magnitud y orientación
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            orientation = np.arctan2(grad_y, grad_x)

            # Dividir en celdas y calcular histogramas
            cell_size = 8
            n_bins = 9
            hog_features = []

            for i in range(0, image.shape[0] - cell_size, cell_size):
                for j in range(0, image.shape[1] - cell_size, cell_size):
                    cell_mag = magnitude[i:i + cell_size, j:j + cell_size]
                    cell_ori = orientation[i:i + cell_size, j:j + cell_size]

                    # Histograma de orientaciones ponderado por magnitud
                    hist, _ = np.histogram(cell_ori, bins=n_bins, range=(-np.pi, np.pi), weights=cell_mag)
                    hog_features.extend(hist / (np.sum(hist) + 1e-7))

            return np.array(hog_features)
        except Exception as e:
            logger.warning(f"Error en HOG: {e}")
            return np.zeros(128)  # Vector por defecto

    def _extract_symmetry_features(self, image: np.ndarray) -> np.ndarray:
        """Características de simetría facial"""
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

                # Estadísticas de simetría
                symmetry_features = [
                    np.mean(symmetry_diff),
                    np.std(symmetry_diff),
                    np.median(symmetry_diff),
                    np.percentile(symmetry_diff, 25),
                    np.percentile(symmetry_diff, 75)
                ]
            else:
                symmetry_features = [0.0] * 5

            return np.array(symmetry_features)
        except Exception as e:
            logger.warning(f"Error en simetría: {e}")
            return np.zeros(5)

    def compare_features(self, features1: np.ndarray, features2: np.ndarray, threshold: float = 0.70) -> dict:
        """Comparación básica (compatibilidad hacia atrás)"""
        if self.enhanced_mode and SKLEARN_AVAILABLE:
            return self.compare_features_enhanced(features1, features2, threshold)
        else:
            return self._compare_features_basic(features1, features2, threshold)

    def compare_features_enhanced(self, features1: np.ndarray, features2: np.ndarray, threshold: float = 0.70) -> dict:
        """Comparación mejorada de características con múltiples métricas"""
        try:
            # Validar entrada
            if len(features1) != len(features2):
                logger.error("Las características no tienen la misma dimensión")
                return self._get_default_comparison(threshold)

            # 1. Similitud coseno (más robusta a escalado)
            if SKLEARN_AVAILABLE:
                cosine_sim = cosine_similarity([features1], [features2])[0][0]
            else:
                # Coseno manual
                dot_product = np.dot(features1, features2)
                norm_product = np.linalg.norm(features1) * np.linalg.norm(features2)
                cosine_sim = dot_product / (norm_product + 1e-7)

            # 2. Correlación de Pearson
            correlation = np.corrcoef(features1, features2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0

            # 3. Distancia euclidiana normalizada
            euclidean_dist = np.linalg.norm(features1 - features2)
            max_possible_dist = np.sqrt(len(features1) * 4)  # Estimación conservadora
            euclidean_similarity = max(0, 1 - (euclidean_dist / max_possible_dist))

            # 4. Distancia de Manhattan normalizada
            manhattan_dist = np.sum(np.abs(features1 - features2))
            max_manhattan_dist = len(features1) * 2  # Estimación para características normalizadas
            manhattan_similarity = max(0, 1 - (manhattan_dist / max_manhattan_dist))

            # 5. Similitud por segmentos (robustez a variaciones locales)
            segment_similarities = []
            segment_size = len(features1) // 8
            for i in range(0, len(features1), segment_size):
                seg1 = features1[i:i + segment_size]
                seg2 = features2[i:i + segment_size]
                if len(seg1) > 0 and len(seg2) > 0:
                    if SKLEARN_AVAILABLE:
                        seg_sim = cosine_similarity([seg1], [seg2])[0][0]
                    else:
                        dot_prod = np.dot(seg1, seg2)
                        norm_prod = np.linalg.norm(seg1) * np.linalg.norm(seg2)
                        seg_sim = dot_prod / (norm_prod + 1e-7)

                    if not np.isnan(seg_sim):
                        segment_similarities.append(seg_sim)

            avg_segment_similarity = np.mean(segment_similarities) if segment_similarities else 0.0

            # Ponderación de métricas (ajustar según rendimiento)
            weights = getattr(settings, 'COMPARISON_WEIGHTS', {
                'cosine': 0.35,
                'correlation': 0.20,
                'euclidean': 0.20,
                'manhattan': 0.15,
                'segments': 0.10
            })

            # Similitud final ponderada
            final_similarity = (
                    weights['cosine'] * cosine_sim +
                    weights['correlation'] * abs(correlation) +
                    weights['euclidean'] * euclidean_similarity +
                    weights['manhattan'] * manhattan_similarity +
                    weights['segments'] * avg_segment_similarity
            )

            # Umbral adaptativo basado en la consistencia de métricas
            metrics = [cosine_sim, abs(correlation), euclidean_similarity, manhattan_similarity]
            consistency = 1.0 - np.std(metrics)  # Mayor consistencia = mayor confianza

            # Ajustar umbral dinámicamente si está habilitado
            if getattr(settings, 'ADAPTIVE_THRESHOLD', True):
                adjusted_threshold = threshold * (0.8 + 0.2 * consistency)
            else:
                adjusted_threshold = threshold

            return {
                "similarity": float(final_similarity),
                "cosine_similarity": float(cosine_sim),
                "correlation": float(correlation),
                "euclidean_similarity": float(euclidean_similarity),
                "manhattan_similarity": float(manhattan_similarity),
                "segment_similarity": float(avg_segment_similarity),
                "consistency": float(consistency),
                "euclidean_distance": float(euclidean_dist),
                "manhattan_distance": float(manhattan_dist),
                "is_match": bool(final_similarity >= adjusted_threshold),
                "threshold": float(threshold),
                "adjusted_threshold": float(adjusted_threshold),
                "confidence": float(final_similarity),
                "metrics_used": list(weights.keys())
            }

        except Exception as e:
            logger.error(f"Error en comparación mejorada: {e}")
            return self._get_default_comparison(threshold)

    def compare_features_with_voting(self, features1: np.ndarray, features2: np.ndarray,
                                     threshold: float = 0.70) -> dict:
        """Comparación con sistema de votación para mayor robustez"""
        try:
            votes = []
            detailed_scores = {}

            # Verificar dimensiones
            if len(features1) != len(features2):
                logger.error("Las características no tienen la misma dimensión")
                return self._get_default_comparison(threshold)

            # Método 1: Similitud coseno
            if SKLEARN_AVAILABLE:
                cosine_sim = cosine_similarity([features1], [features2])[0][0]
            else:
                dot_product = np.dot(features1, features2)
                norm_product = np.linalg.norm(features1) * np.linalg.norm(features2)
                cosine_sim = dot_product / (norm_product + 1e-7)

            votes.append(("cosine", cosine_sim >= threshold))
            detailed_scores["cosine"] = float(cosine_sim)

            # Método 2: Correlación
            correlation = np.corrcoef(features1, features2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            corr_vote = abs(correlation) >= threshold
            votes.append(("correlation", corr_vote))
            detailed_scores["correlation"] = float(abs(correlation))

            # Método 3: Distancia euclidiana normalizada
            euclidean_dist = np.linalg.norm(features1 - features2)
            max_possible_dist = np.sqrt(len(features1) * 4)
            euclidean_sim = max(0, 1 - (euclidean_dist / max_possible_dist))
            votes.append(("euclidean", euclidean_sim >= threshold))
            detailed_scores["euclidean"] = float(euclidean_sim)

            # Método 4: Distancia Manhattan normalizada
            manhattan_dist = np.sum(np.abs(features1 - features2))
            max_manhattan_dist = len(features1) * 2
            manhattan_sim = max(0, 1 - (manhattan_dist / max_manhattan_dist))
            votes.append(("manhattan", manhattan_sim >= threshold))
            detailed_scores["manhattan"] = float(manhattan_sim)

            # Método 5: Embeddings (si están disponibles)
            if self.embedding_extractor and settings.USE_FACE_EMBEDDINGS:
                # Extraer embeddings de las últimas 128 características (si fueron incluidas)
                embedding_size = 128
                if len(features1) >= embedding_size and len(features2) >= embedding_size:
                    embed1 = features1[-embedding_size:]
                    embed2 = features2[-embedding_size:]
                    embedding_sim = self.embedding_extractor.calculate_embedding_distance(embed1, embed2)
                    votes.append(("embeddings", embedding_sim >= threshold))
                    detailed_scores["embeddings"] = float(embedding_sim)

            # Conteo de votos
            positive_votes = sum(1 for _, vote in votes if vote)
            total_votes = len(votes)

            # Decisión por mayoría
            min_votes = settings.MIN_VOTES_REQUIRED if settings.USE_VOTING_SYSTEM else (total_votes / 2)
            is_match = positive_votes >= min_votes

            # Calcular confianza ponderada
            weights = settings.COMPARISON_WEIGHTS
            weighted_sum = 0
            weight_total = 0

            for method, score in detailed_scores.items():
                if method in weights:
                    weighted_sum += score * weights[method]
                    weight_total += weights[method]

            final_confidence = weighted_sum / weight_total if weight_total > 0 else 0

            return {
                "is_match": bool(is_match),
                "confidence": float(final_confidence),
                "voting_confidence": float(positive_votes / total_votes),
                "votes": dict(votes),
                "detailed_scores": detailed_scores,
                "positive_votes": int(positive_votes),
                "total_votes": int(total_votes),
                "min_votes_required": int(min_votes),
                "threshold": float(threshold),
                "method": "voting_system"
            }

        except Exception as e:
            logger.error(f"Error en sistema de votación: {e}")
            return self._get_default_comparison(threshold)

    def _normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """Normalización avanzada de iluminación"""
        try:
            # Convertir a LAB
            lab = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
                               cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Aplicar CLAHE solo al canal L
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)

            # Recombinar
            lab_clahe = cv2.merge([l_clahe, a, b])
            result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

            return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        except Exception as e:
            logger.warning(f"Error en normalización: {e}")
            return image

    def _compare_features_basic(self, features1: np.ndarray, features2: np.ndarray, threshold: float = 0.70) -> dict:
        """Comparación básica original"""
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

            # Convertir todos los valores a tipos nativos de Python
            return {
                "similarity": float(similarity),
                "correlation": float(correlation),
                "distance": float(distance),
                "distance_similarity": float(distance_similarity),
                "is_match": bool(similarity >= threshold),
                "threshold": float(threshold)
            }

        except Exception as e:
            logger.error(f"Error comparando características básicas: {e}")
            return self._get_default_comparison(threshold)

    def _get_default_comparison(self, threshold: float) -> dict:
        """Comparación por defecto en caso de error"""
        return {
            "similarity": 0.0,
            "cosine_similarity": 0.0,
            "correlation": 0.0,
            "euclidean_similarity": 0.0,
            "manhattan_similarity": 0.0,
            "segment_similarity": 0.0,
            "consistency": 0.0,
            "euclidean_distance": float('inf'),
            "manhattan_distance": float('inf'),
            "is_match": False,
            "threshold": float(threshold),
            "adjusted_threshold": float(threshold),
            "confidence": 0.0,
            "metrics_used": []
        }

    def get_processor_info(self) -> dict:
        """Obtener información del procesador"""
        return {
            "enhanced_mode": self.enhanced_mode,
            "sklearn_available": SKLEARN_AVAILABLE,
            "use_dlib": self.use_dlib,
            "opencv_version": cv2.__version__,
            "capabilities": {
                "multiple_detectors": getattr(settings, 'USE_MULTIPLE_DETECTORS', True),
                "adaptive_threshold": getattr(settings, 'ADAPTIVE_THRESHOLD', True),
                "feature_normalization": getattr(settings, 'NORMALIZE_FEATURES', True)
            }
        }


# Funciones de compatibilidad hacia atrás
def get_facial_processor():
    """Obtener instancia del procesador facial"""
    return FacialProcessor()


# Alias para compatibilidad con código existente
ImprovedFacialProcessor = FacialProcessor