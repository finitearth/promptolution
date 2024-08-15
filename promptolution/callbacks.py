from logging import getLogger


class LoggerCallback:
    def __init__(self, logger):
        # TODO check up whats up with logging leves
        self.logger = getLogger(__name__)

    def on_step_end(self, optimizer):
        self.logger.critical(f"Step ended - {'\n'.join(optimizer.prompts)}")
        self.logger.critical(f"Step ended - {[s.item() for s in optimizer.scores]}")
        self.logger.critical(f"Best prompt - {optimizer.prompts[0]}")
        self.logger.critical(f"Best score - {optimizer.scores[0]}")

    def on_epoch_end(self, epoch, logs=None):
        self.logger.critical(f"Epoch {epoch} - {logs}")

    def on_train_end(self, logs=None):
        self.logger.critical(f"Training ended - {logs}")

# TODO callbacks for CSV's, etc. of prompts as well as its scores! Also: Callback to save best prompt over iterations