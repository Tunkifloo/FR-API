import os
import urllib.request
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_file(url, destination):
    """Descargar archivo si no existe"""
    if os.path.exists(destination):
        logger.info(f"✓ Archivo ya existe: {destination}")
        return True

    logger.info(f"⬇ Descargando: {url}")
    logger.info(f"  Destino: {destination}")

    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Descargar con progreso
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            sys.stdout.write(f'\r  Progreso: {percent:.1f}%')
            sys.stdout.flush()

        urllib.request.urlretrieve(url, destination, reporthook=download_progress)
        print()  # Nueva línea después del progreso
        logger.info(f"✓ Descargado exitosamente: {destination}")
        return True
    except Exception as e:
        logger.error(f"✗ Error descargando {url}: {e}")
        return False


def main():
    """Función principal"""
    logger.info("=== Sistema de Reconocimiento Facial - Descarga de Modelos ===")

    # Crear directorio de modelos
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"📁 Directorio de modelos: {models_dir}")

    # Lista de modelos a descargar
    models = [
        {
            "name": "Modelo DNN - Caffe (10.5 MB)",
            "url": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            "file": "res10_300x300_ssd_iter_140000.caffemodel"
        },
        {
            "name": "Configuración DNN - Deploy Prototxt",
            "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "file": "deploy.prototxt"
        },
        {
            "name": "Modelo de Embeddings - OpenFace (31 MB)",
            "url": "https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7",
            "file": "openface_nn4.small2.v1.t7"
        }
    ]

    # Descargar cada modelo
    logger.info("\n📥 Iniciando descarga de modelos...")
    success_count = 0

    for i, model in enumerate(models, 1):
        logger.info(f"\n[{i}/{len(models)}] {model['name']}")
        file_path = os.path.join(models_dir, model['file'])

        if download_file(model['url'], file_path):
            success_count += 1
            # Verificar tamaño del archivo
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"  Tamaño: {size_mb:.1f} MB")

    # Resumen
    logger.info("\n" + "=" * 50)
    logger.info(f"✅ Descarga completada: {success_count}/{len(models)} modelos")

    if success_count == len(models):
        logger.info("\n🎉 ¡Todos los modelos descargados exitosamente!")
        logger.info("\n📝 Para habilitar los modelos avanzados, actualiza tu archivo .env:")
        logger.info("   USE_DNN_DETECTION=true")
        logger.info("   USE_FACE_EMBEDDINGS=true")
        logger.info("\n🚀 El sistema está listo para usar las mejoras de precisión")
    else:
        logger.warning("\n⚠️  Algunos modelos no se pudieron descargar")
        logger.warning("   El sistema funcionará pero sin todas las mejoras")

    logger.info("\n" + "=" * 50)


if __name__ == "__main__":
    main()