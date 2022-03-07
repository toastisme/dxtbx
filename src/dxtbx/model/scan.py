from __future__ import annotations

import os

import pycbf

import libtbx.phil
from scitbx.array_family import flex

from dxtbx.model.scan_helpers import scan_helper_image_files

try:
    from ..dxtbx_model_ext import Scan, ScanBase, TOFSequence
except ModuleNotFoundError:
    from dxtbx_model_ext import Scan, ScanBase, TOFSequence  # type: ignore

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Tuple

scan_phil_scope = libtbx.phil.parse(
    """
  scan
    .expert_level = 1
    .short_caption = "Scan overrides"
  {
    type = *scan tof
      .type = choice
      .help = "Override the scan type"
      .short_caption = "scan_type"

    image_range = None
      .type = ints(size=2)
      .help = "Override the image range"
      .short_caption = "Image range"

    extrapolate_scan = False
      .type = bool
      .help = "When overriding the image range, extrapolate exposure and epoch information from existing images"
      .short_caption = "Extrapolate scan"

    oscillation = None
      .type = floats(size=2)
      .help = "Override the image oscillation"
      .short_caption = "Oscillation"

    batch_offset = None
      .type = int(value_min=0)
      .help = "Override the batch offset"
      .short_caption = "Batch offset"

    tof = None
      .type floats
      .help = "Overrides time-of-flight values"
      .short_caption = "Time-of-flight"

    tof_wavelengths = None
      .type = floats
      .help = "Overrides wavelengths for time-of-flight sequences"
      .short_caption = "Time-of-flight wavelengths"
  }
"""
)


class ScanType(Enum):
    Rotational = 1
    TOF = 2


class AbstractScanFactory(ABC):

    """
    Factory interface for all scan factories
    """

    @staticmethod
    @abstractmethod
    def from_phil(params, reference: ScanBase = None) -> ScanBase:
        """
        Convert phil parameters into a scan model
        """
        ...

    @staticmethod
    @abstractmethod
    def from_dict(dict: Dict, template: Dict) -> ScanBase:
        """
        Convert dictionary to a scan model
        """
        ...

    @staticmethod
    @abstractmethod
    def make_scan(**kwargs) -> ScanBase:
        """
        Convert params into a scan model
        """
        ...


class ScanBaseFactory(AbstractScanFactory):

    """
    Factory to decide between concrete scan factories based on input
    """

    @staticmethod
    def from_phil(params, reference: ScanBase = None) -> ScanBase:
        """
        Convert phil parameters into a scan model
        """

        if params.scan.type == "tof":
            return TOFSequenceFactory.from_phil(params=params, reference=reference)
        else:  # Default to Scan for backwards compatibility
            return ScanFactory(params=params, reference=reference)

    @staticmethod
    def from_dict(dict: Dict, template: Dict) -> ScanBase:
        """
        Convert dictionary to a scan model
        """

        if template is not None:
            if "__id__" in dict and "__id__" in template:
                assert (
                    dict["__id__"] == template["__id__"]
                ), "Scan and template dictionaries are not the same type."

        # Assume dictionaries without "__id__" are for Scan objects
        if "__id__" not in dict or dict["__id__"] == "Rotational":
            return ScanFactory.from_dict(dict=dict, template=template)
        elif dict["__id__"] == "TOF":
            return TOFSequenceFactory.from_dict(dict=dict, template=template)
        else:
            raise NotImplementedError(f"Unknown scan type {dict['__id__']}")

    @staticmethod
    def make_scan(**kwargs) -> ScanBase:
        """
        Convert params into a scan model
        """

        scan_type = kwargs.get("scan_type")

        # Default to scan for back compatability
        if scan_type is None or scan_type == ScanType.Rotational:
            return ScanFactory.make_scan(**kwargs)
        elif scan_type == ScanType.TOF:
            return TOFSequenceFactory.make_scan(**kwargs)
        else:
            raise NotImplementedError(f"Unknown scan type {scan_type}")


class ScanFactory(AbstractScanFactory):
    """A factory for scan instances, to help with constructing the classes
    in a set of common circumstances."""

    @staticmethod
    def from_phil(params, reference: Scan = None) -> Scan:
        """
        Generate a scan model from phil parameters

        """
        if reference is None:
            if params.scan.image_range is None and params.scan.oscillation is None:
                return None
            if params.scan.image_range is None:
                raise RuntimeError("No image range set")
            if params.scan.oscillation is None:
                raise RuntimeError("No oscillation set")
            scan = Scan(params.scan.image_range, params.scan.oscillation)
        else:
            scan = reference

            if params.scan.image_range is not None:
                most_recent_image_index = (
                    scan.get_image_range()[1] - scan.get_image_range()[0]
                )
                scan.set_oscillation(
                    scan.get_image_oscillation(params.scan.image_range[0])
                )
                scan.set_image_range(params.scan.image_range)
                if (
                    params.scan.extrapolate_scan
                    and (params.scan.image_range[1] - params.scan.image_range[0])
                    > most_recent_image_index
                ):
                    exposure_times = scan.get_exposure_times()
                    epochs = scan.get_epochs()
                    exposure_time = exposure_times[most_recent_image_index]
                    epoch_correction = epochs[most_recent_image_index]
                    for i in range(
                        most_recent_image_index + 1,
                        params.scan.image_range[1] - params.scan.image_range[0] + 1,
                    ):
                        exposure_times[i] = exposure_time
                        epoch_correction += exposure_time
                        epochs[i] = epoch_correction
                    scan.set_epochs(epochs)
                    scan.set_exposure_times(exposure_times)

            if params.scan.oscillation is not None:
                scan.set_oscillation(params.scan.oscillation)

        if params.scan.batch_offset is not None:
            scan.set_batch_offset(params.scan.batch_offset)

        return scan

    @staticmethod
    def from_dict(dict: Dict, template: Dict = None) -> Scan:
        """
        Convert the dictionary to a scan model
        """
        if dict is None and template is None:
            return None
        joint = template.copy() if template else {}
        joint.update(dict)

        if not isinstance(joint["exposure_time"], list):
            joint["exposure_time"] = [joint["exposure_time"]]
        joint.setdefault("batch_offset", 0)  # backwards compatibility 20180205
        joint.setdefault("valid_image_ranges", {})  # backwards compatibility 20181113

        # Create the model from the joint dictionary
        return Scan.from_dict(joint)

    @staticmethod
    def make_scan(batch_offset: int = 0, deg: bool = True, **kwargs) -> Scan:
        """
        Convert params into a scan model
        """

        image_range = kwargs.get("image_range")
        exposure_times = kwargs.get("exposure_times")
        oscillation = kwargs.get("oscillation")
        epochs = kwargs.get("epochs")

        if not isinstance(exposure_times, list):
            num_images = image_range[1] - image_range[0] + 1
            exposure_times = [exposure_times for i in range(num_images)]
        else:
            num_images = image_range[1] - image_range[0] + 1
            num_exp = len(exposure_times)
            if num_exp != num_images:
                if num_exp == 0:
                    exposure_times = [0 for i in range(num_images)]
                else:
                    exposure_times = exposure_times.extend(
                        [exposure_times[-1] for i in range(num_images - num_exp)]
                    )

        epoch_list = [epochs[j] for j in sorted(epochs)]

        return Scan(
            tuple(map(int, image_range)),
            tuple(map(float, oscillation)),
            flex.double(list(map(float, exposure_times))),
            flex.double(list(map(float, epoch_list))),
            batch_offset,
            deg,
        )

    @staticmethod
    def single_file(
        filename: str,
        exposure_times: Tuple[float, ...],
        osc_start: float,
        osc_width: float,
        epoch: float,
    ) -> Scan:
        """Construct an scan instance for a single image."""
        index = scan_helper_image_files.image_to_index(os.path.split(filename)[-1])
        if epoch is None:
            epoch = 0.0

        # if the oscillation width is negative at this stage it is almost
        # certainly an artefact of the omega end being 0 when the omega start
        # angle was ~ 360 so it would be ~ -360 - see dxtbx#378
        if osc_width < -180:
            osc_width += 360

        return ScanFactory.make_scan(
            image_range=(index, index),
            exposure_times=exposure_times,
            oscillation=(osc_start, osc_width),
            epochs={index: epoch},
        )

    @staticmethod
    def imgCIF(cif_file: str) -> Scan:
        """Initialize a scan model from an imgCIF file."""

        cbf_handle = pycbf.cbf_handle_struct()
        cbf_handle.read_file(cif_file, pycbf.MSG_DIGEST)

        return ScanFactory.imgCIF_H(cif_file, cbf_handle)

    @staticmethod
    def imgCIF_H(cif_file: str, cbf_handle: pycbf.cbf_handle_struct) -> Scan:
        """Initialize a scan model from an imgCIF file handle, where it is
        assumed that the file has already been read."""

        exposure = cbf_handle.get_integration_time()
        timestamp = cbf_handle.get_timestamp()[0]

        gonio = cbf_handle.construct_goniometer()
        try:
            angles = tuple(gonio.get_rotation_range())
        except Exception as e:
            if str(e).strip() == "CBFlib Error(s): CBF_NOTFOUND":
                # probably a still shot -> no scan object
                return None
            raise

        # xia2-56 handle gracefully reverse turning goniometers - this assumes the
        # rotation axis is correctly inverted in the goniometer factory
        if angles[1] < 0:
            angles = -angles[0], -angles[1]

        index = scan_helper_image_files.image_to_index(cif_file)

        gonio.__swig_destroy__(gonio)

        return ScanFactory.make_scan(
            image_range=(index, index),
            exposure_times=exposure,
            oscillation=angles,
            epochs={index: timestamp},
        )

    @staticmethod
    def add(scans: Tuple[Scan, ...]) -> Scan:
        """Sum a list of scans wrapping the slightly clumsy idiomatic method:
        sum(scans[1:], scans[0])."""
        return sum(scans[1:], scans[0])

    @staticmethod
    def search(filename: str) -> Tuple[str, ...]:
        """Get a list of files which appear to match the template and
        directory implied by the input filename. This could well be used
        to get a list of image headers to read and hence construct scans
        from."""

        template, directory = scan_helper_image_files.image_to_template_directory(
            filename
        )

        indices = scan_helper_image_files.template_directory_to_indices(
            template, directory
        )

        return [
            scan_helper_image_files.template_directory_index_to_image(
                template, directory, index
            )
            for index in indices
        ]


class TOFSequenceFactory(AbstractScanFactory):

    """
    Factory for creating TOFSequences
    """

    @staticmethod
    def from_phil(params, reference: TOFSequence = None) -> TOFSequence:
        """
        Convert phil parameters into a scan model
        """

        def check_for_required_params(params, reference: TOFSequence) -> None:
            if params.scan.tof is None and reference is None:
                raise RuntimeError("Cannot create TOFSequence: tof values are not set")
            if params.scan.tof_wavelengths is None and reference is None:
                raise RuntimeError(
                    "Cannot create ToFSequence: tof_wavelengths are not set"
                )

        def sanity_check_params(params) -> None:
            assert len(params.scan.tof) == len(params.scan.tof_wavelengths)

        check_for_required_params(params=params, reference=reference)
        sanity_check_params(params=params)

        if reference is None:
            scan = TOFSequence
        else:
            scan = reference

        # Even with a reference the new image range must be consistent with new
        # tof/wavelength values
        if params.scan.image_range is not None:
            scan.set_image_range(params.scan.image_range)
        elif reference is None:
            scan.set_image_range((1, len(params.scan.tof)))

        if params.scan.tof is not None and params.scan.tof_wavelengths is not None:
            scan.set_tof_wavelengths(params.scan.tof, params.scan.tof_wavelengths)

        if params.scan.batch_offset is not None:
            scan.set_batch_offset(params.scan.batch_offset)

        return scan

    @staticmethod
    def from_dict(dict: Dict, template: Dict) -> TOFSequence:
        """
        Convert dictionary to a scan model
        """

        required_keys = (
            "tof",
            "wavelengths",
        )

        if dict is None and template is None:
            return None
        joint = template.copy() if template else {}
        joint.update(dict)

        joint.setdefault("batch_offset", 0)  # backwards compatibility 20180205
        joint.setdefault("valid_image_ranges", {})  # backwards compatibility 20181113
        joint.setdefault("image_range", (1, len(joint["tof"])))

        for i in required_keys:
            if i not in joint:
                raise RuntimeError(f"Cannot create TOFSequence: {i} not in dictionary")

        return TOFSequence.from_dict(joint)

    @staticmethod
    def make_scan(**kwargs) -> TOFSequence:
        """
        Convert params into a scan model
        """

        tof = kwargs.get("tof")
        wavelengths = kwargs.get("wavelengths")
        image_range = kwargs.get("image_range")
        batch_offset = kwargs.get("batch_offset")
        if tof is None:
            raise RuntimeError("Cannot create TOFSequence: ToF values not given")
        if wavelengths is None:
            raise RuntimeError("Cannot create TOFSequence: wavelengths not given")
        if image_range is None:
            image_range = (1, len(tof))
        if batch_offset is None:
            batch_offset = 0

        return TOFSequence(
            image_range=tuple(map(int, image_range)),
            tof=tuple(map(float, tof)),
            wavelengths=tuple(map(float, wavelengths)),
            batch_offset=int(batch_offset),
        )
