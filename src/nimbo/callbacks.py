"""Training callbacks for Nimbo - Priority 2 improvement."""

import logging
from typing import Any, Callable, Dict, List, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class NimboCallback(TrainerCallback):
    """Base callback class for Nimbo with common utilities."""

    def __init__(self, name: str = "NimboCallback"):
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class ProgressCallback(NimboCallback):
    """Callback for tracking and displaying training progress."""

    def __init__(
        self,
        log_interval: int = 10,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        super().__init__("ProgressCallback")
        self.log_interval = log_interval
        self.on_progress = on_progress

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return

        progress_info = {
            "step": state.global_step,
            "epoch": state.epoch,
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "eval_loss": logs.get("eval_loss"),
        }

        if self.on_progress:
            self.on_progress(progress_info)
        else:
            logger.info(f"Step {state.global_step}: {logs}")


class EarlyStoppingCallback(NimboCallback):
    """Enhanced early stopping callback with configurable patience."""

    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.0,
        metric: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        super().__init__("EarlyStoppingCallback")
        self.patience = patience
        self.threshold = threshold
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.best_metric: Optional[float] = None
        self.patience_counter = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None or self.metric not in metrics:
            return

        current_metric = metrics[self.metric]

        if self.best_metric is None:
            self.best_metric = current_metric
            self.patience_counter = 0
            return

        if self.greater_is_better:
            improved = current_metric > self.best_metric + self.threshold
        else:
            improved = current_metric < self.best_metric - self.threshold

        if improved:
            self.best_metric = current_metric
            self.patience_counter = 0
            logger.info(
                f"New best {self.metric}: {current_metric:.4f}"
            )
        else:
            self.patience_counter += 1
            logger.info(
                f"No improvement in {self.metric}. "
                f"Patience: {self.patience_counter}/{self.patience}"
            )

            if self.patience_counter >= self.patience:
                logger.info("Early stopping triggered!")
                control.should_training_stop = True


class CheckpointCallback(NimboCallback):
    """Callback for custom checkpoint handling."""

    def __init__(
        self,
        on_save: Optional[Callable[[str, int], None]] = None,
        on_load: Optional[Callable[[str], None]] = None,
    ):
        super().__init__("CheckpointCallback")
        self.on_save = on_save
        self.on_load = on_load

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self.on_save:
            self.on_save(args.output_dir, state.global_step)

    def on_load(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self.on_load:
            self.on_load(args.output_dir)


class MemoryCallback(NimboCallback):
    """Callback for monitoring GPU/MPS memory usage."""

    def __init__(self, log_interval: int = 100):
        super().__init__("MemoryCallback")
        self.log_interval = log_interval

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.global_step % self.log_interval != 0:
            return

        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(
                    f"Step {state.global_step} - "
                    f"GPU Memory: {allocated:.2f}GB allocated, "
                    f"{reserved:.2f}GB reserved"
                )
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS doesn't have detailed memory tracking
                logger.info(f"Step {state.global_step} - MPS device active")
        except Exception as e:
            logger.debug(f"Could not log memory: {e}")


class LossTrackingCallback(NimboCallback):
    """Callback for tracking and storing loss history."""

    def __init__(self):
        super().__init__("LossTrackingCallback")
        self.train_losses: List[float] = []
        self.eval_losses: List[float] = []
        self.steps: List[int] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return

        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.steps.append(state.global_step)

        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])

    def get_history(self) -> Dict[str, List]:
        """Get loss history."""
        return {
            "steps": self.steps,
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
        }


class WandbCallback(NimboCallback):
    """Optional Weights & Biases integration callback."""

    def __init__(
        self,
        project: str = "nimbo",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("WandbCallback")
        self.project = project
        self.run_name = name
        self.config = config
        self._initialized = False

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._initialized:
            return

        try:
            import wandb

            wandb.init(
                project=self.project,
                name=self.run_name,
                config=self.config,
            )
            self._initialized = True
            logger.info(f"W&B run initialized: {wandb.run.name}")
        except ImportError:
            logger.warning("wandb not installed. Skipping W&B logging.")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        if not self._initialized or logs is None:
            return

        try:
            import wandb

            wandb.log(logs, step=state.global_step)
        except Exception as e:
            logger.debug(f"Failed to log to W&B: {e}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._initialized:
            return

        try:
            import wandb

            wandb.finish()
        except Exception as e:
            logger.debug(f"Failed to finish W&B run: {e}")


def create_default_callbacks(
    enable_progress: bool = True,
    enable_memory: bool = False,
    enable_loss_tracking: bool = True,
    early_stopping_patience: int = 0,
) -> List[NimboCallback]:
    """Create a default set of callbacks."""
    callbacks: List[NimboCallback] = []

    if enable_progress:
        callbacks.append(ProgressCallback())

    if enable_memory:
        callbacks.append(MemoryCallback())

    if enable_loss_tracking:
        callbacks.append(LossTrackingCallback())

    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(patience=early_stopping_patience))

    return callbacks
