from logging import getLogger


class LoggerCallback:
    def __init__(self, logger):
        # TODO check up whats up with logging leves
        self.logger = getLogger(__name__)

    def on_step_end(self, optimizer):
        self.logger.critical(f"Step ended - {optimizer.prompts}")

    def on_epoch_end(self, epoch, logs=None):
        self.logger.critical(f"Epoch {epoch} - {logs}")

    def on_train_end(self, logs=None):
        self.logger.critical(f"Training ended - {logs}")
