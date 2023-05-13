from abc import abstractmethod


class PolicyInterface:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __call__(self):
        raise NotImplementedError


class Tweak(PolicyInterface):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self):
        pass