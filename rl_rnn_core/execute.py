from .core.mod_esecutor import Trainer
import numpy as np
def train(
        trainer: Trainer,
        episodes: int,
        mode: str,
        batch_size: int,
        optimizer,
        loss,
        metrics=None,
) -> None:
    """Compila e lancia il training sul Trainer già istanziato."""
    trainer.compile_networks(optimizer, loss, metrics)
    trainer.Train(episodes, mode, batch_size)

def predict(
        model,             # o Trainer|CustomDQNModel
        state: np.ndarray,
) -> np.ndarray:
    """
    Restituisce i Q-values per ogni azione dati gli stati in input.
    Se state è 1D, lo espande a batch di dimensione 1.
    """
    if state.ndim == len(model.input_shape):
        state = np.expand_dims(state, axis=0)
    return model.predict(state)

def test():
    return  3
