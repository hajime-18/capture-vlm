"""各処理段階の共通インターフェース"""

from abc import ABC, abstractmethod

from .config import Config
from .target import TargetDocuments


class Step(ABC):
    """パイプラインの1段階。name と run() を実装する。"""

    name: str = ""

    def __init__(self, config: Config, targets: TargetDocuments):
        self.config = config
        self.targets = targets

    @abstractmethod
    def run(self) -> None:
        ...

    def log(self, msg: str) -> None:
        print(f"  {msg}")