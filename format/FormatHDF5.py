from __future__ import absolute_import, division, print_function

from dxtbx import IncorrectFormatError
from dxtbx.format.FormatFile import FormatFile
from dxtbx.format.FormatMultiImage import FormatMultiImage


class FormatHDF5(FormatMultiImage, FormatFile):
    def __init__(self, image_file, **kwargs):
        if not self.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        FormatMultiImage.__init__(self, **kwargs)
        FormatFile.__init__(self, image_file, **kwargs)

    @staticmethod
    def understand(image_file):
        try:
            with FormatHDF5.open_file(image_file, "rb") as fh:
                return fh.read(8) == b"\211HDF\r\n\032\n"
        except IOError:
            return False
