from __future__ import annotations

from sys import argv
from typing import Tuple

from dxtbx import IncorrectFormatError
from dxtbx.format.FormatHDF5 import FormatHDF5
from dxtbx.model import Detector, Goniometer, PolyBeam, Sequence


class FormatNMX(FormatHDF5):

    """
    Class to read files from NMX
    """

    def __init__(self, image_file):
        if not FormatNMX.understand(image_file):
            raise IncorrectFormatError(self, image_file)

    @staticmethod
    def understand(image_file):
        try:
            return FormatNMX.is_nmx_file(image_file)
        except IOError:
            return False

    @staticmethod
    def is_nmx_file(image_file) -> bool:

        """
        Confirms if image_file is from NMX
        """

        return False

    def get_raw_data(self, index) -> Tuple:

        """
        Returns data of each panel for the ToF histogram bin at index
        """
        pass

    def get_beam(self, idx=None) -> PolyBeam:
        pass

    def get_detector(self, idx=None) -> Detector:
        pass

    def get_goniometer(self, idx=None) -> Goniometer:
        pass

    def get_sequence(self, idx=None) -> Sequence:
        pass


if __name__ == "__main__":
    for arg in argv[1:]:
        print(FormatNMX.understand(arg))
