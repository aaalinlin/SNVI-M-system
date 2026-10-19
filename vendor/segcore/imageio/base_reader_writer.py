

from abc import ABC, abstractmethod
from typing import Tuple, Union, List
import numpy as np


class BaseReaderWriter(ABC):
    @staticmethod
    def _check_all_same(input_list):
        # compare all entries to the first
        for i in input_list[1:]:
            if i != input_list[0]:
                return False
        return True

    @staticmethod
    def _check_all_same_array(input_list):
        # compare all entries to the first
        for i in input_list[1:]:
            if i.shape != input_list[0].shape or not np.allclose(i, input_list[0]):
                return False
        return True

    @abstractmethod
    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        pass

    @abstractmethod
    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        pass

    @abstractmethod
    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        pass