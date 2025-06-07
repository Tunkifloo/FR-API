import numpy as np
from typing import List, Dict, Tuple, Any
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricsAnalyzer:
    """Analizador de métricas para optimización del sistema"""

    def __init__(self):
        self.recognition_results = []
        self.threshold_performance = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})

    def add_recognition_result(self, is_same_person: bool, similarity_score: float,
                               threshold: float, method: str = "standard"):
        """Agregar resultado de reconocimiento para análisis"""
        result = {
            "is_same_person": is_same_person,
            "similarity_score": similarity_score,
            "threshold": threshold,
            "method": method,
            "predicted_match": similarity_score >= threshold
        }

        self.recognition_results.append(result)

        # Actualizar métricas por umbral
        if is_same_person and result["predicted_match"]:
            self.threshold_performance[threshold]["tp"] += 1
        elif is_same_person and not result["predicted_match"]:
            self.threshold_performance[threshold]["fn"] += 1
        elif not is_same_person and result["predicted_match"]:
            self.threshold_performance[threshold]["fp"] += 1
        else:
            self.threshold_performance[threshold]["tn"] += 1

    def calculate_metrics(self, threshold: float = None) -> Dict[str, float]:
        """Calcular métricas de rendimiento"""
        if threshold:
            perf = self.threshold_performance[threshold]
        else:
            # Calcular para todos los resultados
            perf = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
            for result in self.recognition_results:
                if result["is_same_person"] and result["predicted_match"]:
                    perf["tp"] += 1
                elif result["is_same_person"] and not result["predicted_match"]:
                    perf["fn"] += 1
                elif not result["is_same_person"] and result["predicted_match"]:
                    perf["fp"] += 1
                else:
                    perf["tn"] += 1

        # Calcular métricas
        total = sum(perf.values())
        accuracy = (perf["tp"] + perf["tn"]) / total if total > 0 else 0

        precision = perf["tp"] / (perf["tp"] + perf["fp"]) if (perf["tp"] + perf["fp"]) > 0 else 0
        recall = perf["tp"] / (perf["tp"] + perf["fn"]) if (perf["tp"] + perf["fn"]) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # False Accept Rate y False Reject Rate
        far = perf["fp"] / (perf["fp"] + perf["tn"]) if (perf["fp"] + perf["tn"]) > 0 else 0
        frr = perf["fn"] / (perf["fn"] + perf["tp"]) if (perf["fn"] + perf["tp"]) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "far": far,  # False Accept Rate
            "frr": frr,  # False Reject Rate
            "true_positives": perf["tp"],
            "false_positives": perf["fp"],
            "true_negatives": perf["tn"],
            "false_negatives": perf["fn"],
            "total_samples": total
        }

    def find_optimal_threshold(self, target_metric: str = "f1_score") -> Tuple[float, Dict[str, float]]:
        """Encontrar umbral óptimo basado en métrica objetivo"""
        best_threshold = 0.5
        best_score = 0
        best_metrics = {}

        # Probar diferentes umbrales
        for threshold in np.arange(0.3, 0.95, 0.05):
            metrics = self.calculate_metrics(threshold)

            if metrics[target_metric] > best_score:
                best_score = metrics[target_metric]
                best_threshold = threshold
                best_metrics = metrics

        logger.info(f"Umbral óptimo: {best_threshold} con {target_metric}={best_score:.4f}")
        return best_threshold, best_metrics

    def generate_report(self) -> Dict[str, Any]:
        """Generar reporte completo de análisis"""
        if not self.recognition_results:
            return {"error": "No hay resultados para analizar"}

        # Estadísticas generales
        similarities = [r["similarity_score"] for r in self.recognition_results]
        same_person_sims = [r["similarity_score"] for r in self.recognition_results if r["is_same_person"]]
        diff_person_sims = [r["similarity_score"] for r in self.recognition_results if not r["is_same_person"]]

        report = {
            "total_comparisons": len(self.recognition_results),
            "similarity_stats": {
                "overall": {
                    "mean": np.mean(similarities),
                    "std": np.std(similarities),
                    "min": np.min(similarities),
                    "max": np.max(similarities)
                },
                "same_person": {
                    "mean": np.mean(same_person_sims) if same_person_sims else 0,
                    "std": np.std(same_person_sims) if same_person_sims else 0,
                    "count": len(same_person_sims)
                },
                "different_person": {
                    "mean": np.mean(diff_person_sims) if diff_person_sims else 0,
                    "std": np.std(diff_person_sims) if diff_person_sims else 0,
                    "count": len(diff_person_sims)
                }
            },
            "current_metrics": self.calculate_metrics(),
            "optimal_thresholds": {}
        }

        # Encontrar umbrales óptimos para diferentes métricas
        for metric in ["f1_score", "accuracy", "precision", "recall"]:
            threshold, metrics = self.find_optimal_threshold(metric)
            report["optimal_thresholds"][metric] = {
                "threshold": threshold,
                "metrics": metrics
            }

        return report