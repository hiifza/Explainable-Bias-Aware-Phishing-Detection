"""
src/utils/metrics_logger.py
---------------------------
Training metrics logging utilities.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsLogger:

    def __init__(self, model_name: str, track: str):
        self.model_name = model_name
        self.track = track

    def log_training_time(self, seconds: float) -> None:
        logger.info(
            f"[{self.model_name}] Training Time: {seconds:.3f}s"
        )

    def log_metrics(
        self,
        metrics: dict,
        stage: str = "test",
        n_samples: int | None = None,
    ) -> None:

        logger.info(
            f"[{self.model_name}] {stage.upper()} Metrics"
        )

        for key, value in metrics.items():

            if key == "confusion_matrix":
                continue

            logger.info(
                f"    {key}: {value}"
            )

        if n_samples is not None:
            logger.info(
                f"    samples: {n_samples}"
            )

    def log_cv(
        self,
        cv_result: dict,
        n_splits: int = 5,
    ) -> None:

        logger.info(
            f"[{self.model_name}] Cross Validation ({n_splits} folds)"
        )

        for key, value in cv_result.items():

            if isinstance(value, (int, float)):
                logger.info(
                    f"    {key}: {value}"
                )

    def log_summary(
        self,
        result: dict,
    ) -> None:

        logger.info(
            f"[{self.model_name}] Summary"
        )

        logger.info(
            f"Accuracy : {result.get('accuracy', 'N/A')}"
        )

        logger.info(
            f"F1 Score : {result.get('f1', 'N/A')}"
        )

        logger.info(
            f"ROC AUC  : {result.get('roc_auc', 'N/A')}"
        )


def log_benchmark_table(df) -> None:

    print("\n" + "=" * 80)
    print("MODEL BENCHMARK RESULTS")
    print("=" * 80)

    try:
        print(df.to_string(index=False))
    except Exception:
        print(df)

    print("=" * 80)