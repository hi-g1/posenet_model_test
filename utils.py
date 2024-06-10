from distutils.util import strtobool

# logger
class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))