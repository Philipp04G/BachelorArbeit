import abc


class RawDataReader(abc.ABC):

    @abc.abstractmethod
    def next_time(self):
        raise NotImplemented

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplemented

    @abc.abstractmethod
    def __next__(self):
        raise NotImplemented

