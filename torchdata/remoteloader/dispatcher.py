class Dispatcher:
    def __init__(self, remote: Remote, path: str, **kwargs):
        self.remote = remote
        self.path = path
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.remote.dispatch(self.path, *args, **kwargs)
    