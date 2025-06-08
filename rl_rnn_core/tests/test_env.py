# tests/test_env.py
import pytest
import numpy as np
from ..core.flex_envoirment import EnvFlex

@pytest.fixture
def dummy_data():
    # crea un DataFrame semplice per testare EnvFlex
    import pandas as pd
    df = pd.DataFrame({
        "price": [100, 101, 102, 99, 98],
        "indicator": [1, 0, 1, 0, 1]
    })
    return df

def test_env_reset(dummy_data):
    env = EnvFlex(data=dummy_data, window_size=3)
    obs = env.reset()
    # deve ritornare un array numpy di forma (3,numero_colonne)
    assert isinstance(obs, np.ndarray)
    assert obs.shape[0] == 3

def test_env_step(dummy_data):
    env = EnvFlex(data=dummy_data, window_size=3)
    _ = env.reset()
    action = 1  # prendi un’azione qualsiasi ammessa
    next_obs, reward, done, info = env.step(action)
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(reward, float) or isinstance(reward, (int,))
    assert isinstance(done, bool)
    assert isinstance(info, dict)
