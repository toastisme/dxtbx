from __future__ import annotations

from dxtbx.format.Format import abstract
from dxtbx.format.FormatMultiImage import FormatMultiImage
from dxtbx.imageset import ImageSetType


@abstract
class FormatMultiImageLazy(FormatMultiImage):
    """
    Lazy version of FormatMultiImage that does not instantiate the models ahead of time.
    It creates an ImageSetLazy class and returns it. Saves time when image file contains
    too many images to setup before processing.
    """

    @classmethod
    def get_imageset(
        Class,
        filenames,
        beam=None,
        detector=None,
        goniometer=None,
        sequence=None,
        imageset_type=ImageSetType.ImageSetLazy,
        single_file_indices=None,
        format_kwargs=None,
    ):

        return super().get_imageset(
            filenames=filenames,
            beam=beam,
            detector=detector,
            goniometer=goniometer,
            sequence=sequence,
            imageset_type=imageset_type,
            single_file_indices=single_file_indices,
            format_kwargs=format_kwargs,
        )
