import abc
from samson_data_reader.datacalsses import *


class ProcessedDataReader(abc.ABC):


    @abc.abstractmethod
    def add_data(self, image_data: ImageData):
        raise NotImplemented