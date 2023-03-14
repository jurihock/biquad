from datetime import timedelta
from timeit import default_timer as timer


class measure(object):

    def __init__(self, name=None):

        self.name = name

    def __enter__(self):

        self.start = timer()

    def __exit__(self, type, value, traceback):

        self.stop = timer()
        self.delta = self.stop - self.start

        delta = timedelta(seconds=self.delta)

        if self.name:
            print(self.name, delta)
        else:
            print(delta)
