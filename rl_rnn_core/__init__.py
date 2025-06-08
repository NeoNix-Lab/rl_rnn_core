if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    main()

from .execute import train
from .execute import predict
from .execute import test

__all__ = ["train", "predict", "test"]
