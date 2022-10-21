import zlib, pickle, pickletools
import diskcache

class CompressedDisk(diskcache.Disk):
    '''
    This is a subclass of diskcache.Disk that stores keys and values by compressing (using zlib)
    their pickled representations.
    '''
    def __init__(self, directory, compress_level=3, **kwargs):
        self.compress_level = compress_level
        super().__init__(directory, **kwargs)

    def _loads(self, raw):
        return pickle.loads(zlib.decompress(raw))
    def _dumps(self, data):
        data = pickle.dumps(data, protocol=self.pickle_protocol)
        # This is also a departure from the default Disk; it only optimizes keys, not values
        # https://github.com/grantjenks/python-diskcache/commit/0bd9d611d05a47f56ab13d95fc9cac1d2775fcc9
        data = pickletools.optimize(data)
        return zlib.compress(data, self.compress_level)

    # Serializing keys
    def put(self, key):
        return super().put(self._dumps(key))
    def get(self, key, raw):
        return self._loads(super().get(key, raw))

    # Serializing data
    def store(self, value, read, key=diskcache.UNKNOWN):
        if not read:
            value = self._dumps(value)
        return super().store(value, read, key=key)
    def fetch(self, mode, filename, value, read):
        data = super().fetch(mode, filename, value, read)
        if not read:
            data = self._loads(data)
        return data
