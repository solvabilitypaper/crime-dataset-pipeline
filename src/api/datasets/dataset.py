from abc import ABC, abstractmethod
from schema import Schema

from pandas import DataFrame


class Dataset(ABC):
    identifier: str
    data: DataFrame or dict

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def to_schema(self, schema: Schema) -> (list or dict):
        pass

    def is_empty(self) -> bool:
        if isinstance(self.data, dict) and not self.data:
            return True

        if isinstance(self.data, DataFrame) and self.data.empty:
            return True

        return False
