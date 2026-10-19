
from polarity_d import trainer as _application_trainers


def __getattr__(name):
    return getattr(_application_trainers, name)
