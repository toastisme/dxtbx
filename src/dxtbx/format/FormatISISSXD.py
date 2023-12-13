from __future__ import annotations

from shutil import copy
from sys import argv

import h5py
import numpy as np
from scipy.constants import Planck, m_n

import cctbx.array_family.flex

from dials.array_family import flex

from dxtbx import IncorrectFormatError
from dxtbx.format.FormatNXTOFRAW import FormatNXTOFRAW
from dxtbx.model import Detector
from dxtbx.model.beam import PolyBeamFactory
from dxtbx.model.goniometer import GoniometerFactory
from dxtbx.model.sequence import SequenceFactory


class FormatISISSXD(FormatNXTOFRAW):

    """
    Class to read NXTOFRAW files from the ISIS SXD
    (https://www.isis.stfc.ac.uk/Pages/sxd.aspx)

    """

    def __init__(self, image_file, **kwargs):
        super().__init__(image_file)
        if not FormatISISSXD.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        self.image_file = image_file
        self.nxs_file = self.open_file(image_file)
        self.detector = None
        self.raw_data = None

    def open_file(self, image_file):
        return h5py.File(image_file, "r")

    @staticmethod
    def understand(image_file):
        try:
            return FormatISISSXD.is_isissxd_file(image_file)
        except (IOError, KeyError):
            return False

    @staticmethod
    def is_isissxd_file(image_file):

        """
        Confirms if image_file is a NXTOFRAW format
        and from the SXD by confirming required fields
        are present and then checking the name attribute

        """

        def get_name(image_file):
            with h5py.File(image_file, "r") as handle:
                if "raw_data_1" in handle:
                    return handle["/raw_data_1/name"][0].decode()
                return ""

        if not FormatNXTOFRAW.understand(image_file):
            return False

        return get_name(image_file) == "SXD"

    def get_spectra_idx_1D(self, panel: int, x_px: int, y_px: int) -> int:

        image_size = self._get_panel_size_in_px()
        total_pixels = image_size[0] * image_size[1]
        # Index offset in SXD data
        # See p24 of https://www.isis.stfc.ac.uk/Pages/sxd-user-guide6683.pdf
        idx_offset = 4
        panel_idx = (y_px * image_size[1]) + x_px
        panel_start_idx = (total_pixels * panel) + (idx_offset * panel)
        return int(panel_start_idx + panel_idx)

    def get_raw_spectra(self, normalize_by_proton_charge=False):
        if normalize_by_proton_charge:
            proton_charge = self.nxs_file["raw_data_1"]["proton_charge"][0]
            return (
                self.nxs_file["raw_data_1"]["detector_1"]["counts"][:] / proton_charge
            )
        return self.nxs_file["raw_data_1"]["detector_1"]["counts"][:]

    def save_spectra(self, spectra, output_filename):
        copy(self.image_file, output_filename)
        nxs_file = h5py.File(output_filename, "r+")
        del nxs_file["raw_data_1"]["detector_1"]["counts"]
        nxs_file["raw_data_1/detector_1"].create_dataset(
            "counts", spectra.shape, dtype=np.dtype("f4"), compression="lzf"
        )
        nxs_file["raw_data_1/detector_1/counts"][:] = spectra
        nxs_file.close()

    def get_instrument_name(self):
        return "SXD"

    def get_experiment_description(self):
        title = self.nxs_file["raw_data_1"]["title"][0].decode()
        run_number = self.nxs_file["raw_data_1"]["run_number"][0]
        return f"{title} ({run_number})"

    def load_raw_data(self, as_numpy_arrays=False, normalise_by_proton_charge=False):
        def get_detector_idx_array(detector_number, image_size, idx_offset):
            total_pixels = image_size[0] * image_size[1]
            min_range = (total_pixels * (detector_number - 1)) + (
                idx_offset * (detector_number - 1)
            )
            max_range = min_range + total_pixels
            return np.arange(min_range, max_range).reshape(image_size).T

        dataset = "raw_data_1"
        raw_counts = self.nxs_file[dataset]["detector_1"]["counts"][0, :, :]

        if normalise_by_proton_charge:
            proton_charge = self.nxs_file[dataset]["proton_charge"][0]
            raw_counts = raw_counts / proton_charge

        num_panels = self._get_num_panels()
        image_size = self._get_panel_size_in_px()
        num_bins = len(self.get_tof_in_seconds())

        # Index offset in SXD data
        # See p24 of https://www.isis.stfc.ac.uk/Pages/sxd-user-guide6683.pdf
        idx_offset = 4
        raw_data = []

        for n in range(1, num_panels + 1):
            idx_array = get_detector_idx_array(n, image_size, idx_offset)
            panel_array = np.zeros((idx_array.shape[0], idx_array.shape[1], num_bins))
            for c_i, i in enumerate(idx_array):
                for c_j, j in enumerate(i):
                    panel_array[c_i, c_j, :] = raw_counts[j, :]
            if as_numpy_arrays:
                raw_data.append(panel_array)
            else:
                flex_array = flex.double(np.ascontiguousarray(panel_array))
                flex_array.reshape(flex.grid(panel_array.shape))
                raw_data.append(flex_array)

        return tuple(raw_data)

    def get_raw_data(self, index):

        if self.raw_data is None:
            self.raw_data = self.load_raw_data()

        raw_data_idx = []
        for i in self.raw_data:
            arr = i[:, :, index : index + 1]
            arr.reshape(flex.grid(i.all()[0], i.all()[1]))
            raw_data_idx.append(arr)
        return tuple(raw_data_idx)

    def get_image_data_2d(self):
        self.raw_data = self.load_raw_data(as_numpy_arrays=True)
        raw_summed_data = []
        max_val = None
        for idx, i in enumerate(self.raw_data):
            arr = np.sum(i, axis=2).T
            if idx != 0:
                arr = np.flipud(arr)
            arr_max_val = np.max(arr)
            if max_val is None or arr_max_val > max_val:
                max_val = arr_max_val
            raw_summed_data.append(arr.flatten())
        return tuple([(i / max_val).tolist() for i in raw_summed_data])

    def _get_panel_pixel_s1(self, detector, center=True):
        def get_panel_pixels_in_mm_as_1d(flip):
            pixel_size = self._get_panel_pixel_size_in_mm()
            panel_size = self._get_panel_size_in_px()
            pixels = flex.vec2_double(panel_size[0] * panel_size[1])

            if center:
                count = 0
                for i in range(panel_size[0]):
                    for j in range(panel_size[1]):
                        pixels[count] = (
                            (i * pixel_size[0]) + (pixel_size[0] * 0.5),
                            (j * pixel_size[1]) + (pixel_size[1] * 0.5),
                        )
                        count += 1
            else:
                count = 0
                for i in range(panel_size[0]):
                    for j in range(panel_size[1]):
                        pixels[count] = ((i * pixel_size[0]), (j * pixel_size[1]))
                        count += 1
            return pixels

        pixels_in_mm = get_panel_pixels_in_mm_as_1d(center)

        panel_pixel_s1 = []
        for panel in detector:
            panel_pixel_s1.append(panel.get_lab_coord(pixels_in_mm))

        return panel_pixel_s1

    def get_raw_spectra_two_theta(self, detector, beam):
        unit_s0 = np.array(beam.get_unit_s0())
        panel_pixel_s1 = self._get_panel_pixel_s1(detector, center=False)

        # Index offset in SXD data
        # See p24 of https://www.isis.stfc.ac.uk/Pages/sxd-user-guide6683.pdf
        idx_offset = 4
        raw_spectra_two_theta = []
        for s1_p in panel_pixel_s1:
            s1_p_n = s1_p / s1_p.norms()
            c = np.dot(s1_p_n, unit_s0)
            c = np.clip(c, -1, 1)
            two_theta = np.arccos(c)
            raw_spectra_two_theta += list(two_theta)
            raw_spectra_two_theta += [0] * idx_offset

        return raw_spectra_two_theta

    def get_raw_spectra_L1(self, detector):

        panel_pixel_s1 = self._get_panel_pixel_s1(detector, center=False)

        # Index offset in SXD data
        # See p24 of https://www.isis.stfc.ac.uk/Pages/sxd-user-guide6683.pdf
        idx_offset = 4
        L1 = []
        for s1_p in panel_pixel_s1:
            L1_p = s1_p.norms() * 10**-3
            L1 += list(L1_p)
            L1 += [0] * idx_offset
        return L1

    def _get_detector(self):

        """
        Returns a  Detector instance with parameters taken from

        """

        num_panels = self._get_num_panels()
        panel_names = self._get_panel_names()
        panel_type = self._get_panel_type()
        image_size = self._get_panel_size_in_px()
        trusted_range = self._get_panel_trusted_range()
        pixel_size = self._get_panel_pixel_size_in_mm()
        fast_axes = self._get_panel_fast_axes()
        slow_axes = self._get_panel_slow_axes()
        panel_origins = self._get_panel_origins()
        gain = self._get_panel_gain()
        panel_projections = self._get_panel_projections_2d()
        detector = Detector()
        root = detector.hierarchy()

        for i in range(num_panels):
            panel = root.add_panel()
            panel.set_type(panel_type)
            panel.set_name(panel_names[i])
            panel.set_image_size(image_size)
            panel.set_trusted_range(trusted_range)
            panel.set_pixel_size(pixel_size)
            panel.set_local_frame(fast_axes[i], slow_axes[i], panel_origins[i])
            panel.set_gain(gain)
            r, t = panel_projections[i + 1]
            r = tuple(map(int, r))
            t = tuple(map(int, t))
            panel.set_projection_2d(r, t)

        return detector

    """
    Hardcoded values not contained in the self.nxs_file are taken from
    https://doi.org/10.1107/S0021889806025921
    """

    def _get_time_channel_bins(self):
        return self.nxs_file["raw_data_1"]["instrument"]["dae"]["time_channels_1"][
            "time_of_flight"
        ][:]

    def get_time_channel_bin_widths_in_seconds(self):
        bins = self._get_time_channel_bins()
        return [(bins[i + 1] - bins[i]) * 10**-6 for i in range(len(bins) - 1)]

    def _get_time_channels_in_seconds(self):
        bins = self._get_time_channel_bins()
        return [(bins[i] + bins[i + 1]) * 0.5 * 10**-6 for i in range(len(bins) - 1)]

    def _get_time_channels_in_usec(self):
        bins = self._get_time_channel_bins()
        return [(bins[i] + bins[i + 1]) * 0.5 for i in range(len(bins) - 1)]

    def get_tof_in_seconds(self):
        return self._get_time_channels_in_seconds()

    def get_tof_range(self):
        return (0, len(self.get_tof_in_seconds()))

    def get_wavelength_channels_in_ang(self):
        time_channels = self._get_time_channels_in_seconds()
        L = self._get_sample_to_moderator_distance() * 10**-3
        return [self.get_tof_wavelength_in_ang(L, i) for i in time_channels]

    def get_wavelength_channels(self):
        time_channels = self._get_time_channels_in_seconds()
        L = self._get_sample_to_moderator_distance() * 10**-3
        return [self.get_tof_wavelength(L, i) for i in time_channels]

    def get_wavelength_channels_in_A(self):
        wavelengths = self.get_wavelength_channels()
        return [i * 10**10 for i in wavelengths]

    def get_duration_in_uA(self):
        return self.nxs_file["raw_data_1"]["collection_time"][0]

    def get_gutmann_profile_params(self):

        """
        Taken from SXDII_corrected.instr
        Parameters for profile described in
        https://doi.org/10.1016/j.nima.2016.12.026

        dt/t: eq (7)
        dtheta: eq (7)
        alpha: eq (6)
        beta: eq (6)
        beta_w: wavelength-dependent verstion of beta
        """

        return {
            "dt/t": 0.008,
            "dtheta": 1.000,
            "alpha": 2.000,
            "beta": 0.030,
            "beta_w": 0.015,
        }

    def _get_sample_to_moderator_distance(self):
        return 8300

    def _get_wavelength_range(self):
        return (0.2, 10)

    def _get_num_panels(self):
        return 11

    def _get_panel_names(self):
        return ["%02d" % (i + 1) for i in range(11)]

    def _get_panel_origin_l2_vals_in_mm(self):
        return (
            262.787,
            262.787,
            262.787,
            262.787,
            262.787,
            302.212,
            302.212,
            302.212,
            302.212,
            302.212,
            311.178,
        )

    def _get_panel_gain(self):
        return 1.0

    def _get_panel_trusted_range(self):
        return (-1, 100000)

    def _panel_0_params(self):
        if self._panel_0_flipped():
            return {
                "slow_axis": (0.793, 0.0, 0.609),
                "fast_axis": (0.0, -1.0, 0.0),
                "origin": (60.81, 96.0, -236.946),
            }
        return {
            "fast_axis": (-0.793, 0.0, -0.609),
            "slow_axis": (0.0, 1.0, 0.0),
            "origin": (213.099, -96.0, -120.041),
        }

    def _get_panel_origins(self):
        return (
            self._panel_0_params()["origin"],
            (224.999, -96.0, 96.0),
            (60.809, -96.0, 236.945),
            (-214.172, -96.0, 118.198),
            (-224.999, -96.0, -96.0),
            (-60.809, -96.0, -236.945),
            (127.534, -256.614, 96.0),
            (-96.0, -256.614, 127.534),
            (-123.036, -258.801, -96.0),
            (96.0, -256.614, -127.534),
            (96.0, -278.0, 96.0),
        )

    def _panel_0_flipped(self):
        if self.nxs_file["raw_data_1"]["run_number"][0] > 30000:
            return True
        return False

    def _get_panel_slow_axes(self):
        return (
            self._panel_0_params()["slow_axis"],
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.695, 0.719, -0.0),
            (0.0, 0.719, 0.695),
            (-0.707, 0.707, -0.0),
            (0.0, 0.719, -0.695),
            (-0.0, 0.0, -1.0),
        )

    def _get_panel_fast_axes(self):
        return (
            self._panel_0_params()["fast_axis"],
            (-0.0, -0.0, -1.0),
            (0.793, -0.0, -0.609),
            (0.788, -0.0, 0.616),
            (-0.0, -0.0, 1.0),
            (-0.793, -0.0, 0.609),
            (0.0, -0.0, -1.0),
            (1.0, -0.0, -0.0),
            (-0.0, -0.0, 1.0),
            (-1.0, -0.0, -0.0),
            (-1.0, -0.0, -0.0),
        )

    def _get_s0(self):
        return (0, 0, 0)

    def _get_unit_s0(self):
        return (0, 0, 1)

    def _get_sample_to_source_direction(self):
        return (0, 0, -1)

    def _get_beam_polarization_normal(self):
        return (0, 0, 0)

    def _get_beam_polarization_fraction(self):
        return 0.5

    def _get_beam_flux(self):
        return 0.0

    def _get_beam_transmission(self):
        return 1.0

    def _get_beam_divergence(self):
        return 0.0

    def _get_beam_sigma_divergence(self):
        return 0.0

    def get_num_images(self):
        return len(self.get_tof_in_seconds())

    def get_beam(self, idx=None):
        sample_to_source_dir = self._get_sample_to_source_direction()
        sample_to_mod_d = self._get_sample_to_moderator_distance()
        wavelength_range = self._get_wavelength_range()
        return PolyBeamFactory.make_beam(
            sample_to_source_direction=sample_to_source_dir,
            sample_to_moderator_distance=sample_to_mod_d,
            wavelength_range=wavelength_range,
        )

    def get_detector(self, idx=None):
        return self._get_detector()

    def get_sequence(self, idx=None):
        image_range = (1, self.get_num_images())
        tof_in_seconds = self.get_tof_in_seconds()
        return SequenceFactory.make_tof_sequence(
            image_range=image_range,
            tof_in_seconds=tof_in_seconds,
            wavelengths=self.get_wavelength_channels_in_A(),
        )

    def get_goniometer(self, idx=None):
        rotation_axis = (0.0, 1.0, 0.0)
        fixed_rotation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        return GoniometerFactory.make_goniometer(rotation_axis, fixed_rotation)

    def get_panel_size_in_px(self):
        return (64, 64)

    def _get_panel_size_in_px(self):
        return (64, 64)

    def _get_panel_pixel_size_in_mm(self):
        return (3, 3)

    def _get_panel_size_in_mm(self):
        size_in_px = self._get_panel_size_in_px()
        pixel_size_in_mm = self._get_panel_pixel_size_in_mm()
        return tuple(
            [size_in_px[i] * pixel_size_in_mm[i] for i in range(len(size_in_px))]
        )

    def _get_panel_type(self):
        return "SENSOR_PAD"

    def _get_raw_spectra_array(self):
        # Returns 2D array of (pixels, time_channels) for all 11 detectors
        return self.nxs_file["raw_data_1"]["detector_1"]["counts"][:][0]

    def _get_panel_images(self):

        """
        Returns a list of arrays (x_num_px, y_num_px, num_time_channels)
        for each panel, ordered from 1-11
        """
        raw_data = self._get_raw_spectra_array()

        # Panel positions are offset by 4 in raw_data array
        # See p24 of https://www.isis.stfc.ac.uk/Pages/sxd-user-guide6683.pdf
        panel_size = self._get_panel_size_in_px()
        total_px = panel_size[0] * panel_size[1]
        offsets = [
            (((total_px * i) + (i * 4)), ((total_px * (i + 1)) + (i * 4)))
            for i in range(11)
        ]
        panel_raw_data = [raw_data[i[0] : i[1], :] for i in offsets]

        panel_size = self._get_panel_size_in_px()
        time_channel_size = len(self._get_time_channels_in_seconds())
        array_shape = (panel_size[0], panel_size[1], time_channel_size)
        return [i.reshape(array_shape) for i in panel_raw_data]

    def get_tof_wavelength(self, L, tof):
        return (Planck * tof) / (m_n * L)

    def get_tof_wavelength_in_ang(self, L, tof):
        return self.get_tof_wavelength(L, tof) * 10**10

    def get_max_slice_index(self):
        return len(self.get_tof_in_seconds()) - 1

    def get_pixel_spectra(self, panel_idx, x, y):
        if self.raw_data is None:
            self.raw_data = self.load_raw_data()

        time_channels = self._get_time_channels_in_usec()
        return time_channels, list(self.raw_data[panel_idx][x : x + 1, y : y + 1, :])

    def _get_panel_projections_2d(self) -> dict:

        """
        Returns a projection of the
        detector flattened around the bottom panel (11), with
        all panels facing upright.

        """

        p_w, p_h = self._get_panel_size_in_px()
        panel_pos = {
            11: ((1, 0, 0, 1), (0, 0)),
            10: ((1, 0, 0, 1), (-p_h, 0)),
            8: ((1, 0, 0, 1), (p_h, 0)),
            7: ((1, 0, 0, 1), (0, -p_w)),
            9: ((1, 0, 0, 1), (0, p_w)),
            2: ((1, 0, 0, 1), (0, 2 * -p_w)),
            5: ((1, 0, 0, 1), (0, 2 * p_w)),
            3: ((1, 0, 0, 1), (p_h, 2 * -p_w)),
            4: ((1, 0, 0, 1), (p_h, 2 * p_w)),
            1: ((1, 0, 0, 1), (-p_h, 2 * -p_w)),
            6: ((1, 0, 0, 1), (-p_h, 2 * p_w)),
        }

        return panel_pos

    def get_spectra_L1s(self, detector):

        pixel_size = self._get_panel_pixel_size_in_mm()
        panel_size = self._get_panel_size_in_px()
        num_pixels = panel_size[0] * panel_size[1]
        pixels = cctbx.array_family.flex.vec2_double(num_pixels)
        offset = (0, 0, 0, 0)
        count = 0
        for i in range(panel_size[0]):
            for j in range(panel_size[1]):
                pixels[count] = (i * pixel_size[0], j * pixel_size[1])

        spectra_L1 = []
        for p in range(self._get_num_panels()):
            s1 = detector[p].get_lab_coord(pixels)
            L1 = s1.norms() * 10**-3
            L1 = np.append(L1, offset)
            spectra_L1 = np.append(spectra_L1, L1)

        return spectra_L1

    def get_bin_width_correction(self):
        tof_bins = self._get_time_channel_bins()
        tof_bin_widths = np.abs(np.diff(tof_bins))
        tof_bin_widths /= np.min(tof_bin_widths)
        return tof_bin_widths

    def get_reflection_table_from_use_file(self, use_file, specific_panel=None):

        import dials_array_family_flex_ext
        from scipy import interpolate

        def is_data_row(row):
            num_data_columns = 13
            return len(row.split()) == num_data_columns

        def is_active_row(count, lines):
            num_active_row_columns = 9
            active_pos = -1
            if len(lines[count + 1].split()) == num_active_row_columns:
                return lines[count + 1].split()[active_pos] == "1"
            return False

        def get_hkl(row):
            hkl_pos = (1, 4)
            assert is_data_row(row), "Cannot extract data from this row"
            hkl = row.split()[hkl_pos[0] : hkl_pos[1]]
            return tuple(map(int, hkl))

        def get_single_float_value(row, idx):
            # assert(is_data_row(row)), "Cannot extract data from this row"
            return float(row.split()[idx])

        def get_x(row):
            x_pos = 4
            return get_single_float_value(row, x_pos)

        def get_dx(row):
            dx_pos = 3
            return get_single_float_value(row, dx_pos)

        def get_y(row):
            y_pos = 5
            return get_single_float_value(row, y_pos)

        def get_dy(row):
            dy_pos = 4
            return get_single_float_value(row, dy_pos)

        def get_tof_curve_coefficients(tof_vals):
            x = [i + 1 for i in range(len(tof_vals))]
            return interpolate.splrep(tof_vals, x)

        def get_tof_frame(tof, tof_curve_coeffs):
            return float(interpolate.splev(tof, tof_curve_coeffs))

        def get_bbox(x, dx, y, dy, tof, dtof, tof_curve_coeffs):
            frame = get_tof_frame(tof, tof_curve_coeffs)
            return (
                int(x - dx),
                int(x + dx),
                int(y - dy),
                int(y + dy),
                int(frame - dtof),
                int(frame + dtof),
            )

        def get_tof(row):
            tof_pos = 6
            return get_single_float_value(row, tof_pos)

        def get_dtof(row):
            dtof_pos = 5
            return get_single_float_value(row, dtof_pos)

        def get_pixel_wavelength_in_ang(
            x, y, tof, L0, centroid_l, pixel_size_in_mm, panel_size_in_px
        ):
            import numpy as np

            rel_x = abs(x) * pixel_size_in_mm[0] * 10**-3
            rel_y = abs(y) * pixel_size_in_mm[1] * 10**-3
            rel_pos = np.sqrt(np.square(rel_x) + np.square(rel_y))
            rel_L = np.sqrt(np.square(rel_pos) + np.square(centroid_l))

            return self.get_tof_wavelength_in_ang(L0 + rel_L, tof)

        def get_panel_l(panel_idx):
            panel_l_vals = [
                0.225,
                0.225,
                0.225,
                0.225,
                0.225,
                0.225,
                0.270,
                0.270,
                0.270,
                0.270,
                0.278,
            ]
            return panel_l_vals[panel_idx]

        def at_start_of_table(row):
            return row.startswith(" NSEQ  ")

        def at_end_of_table(row):
            return row.startswith("DETECTOR CALIBRATION FILE ")

        def convert_coord_to_dials(x, y):
            offset = 32
            return (x + offset, y + offset)

        def get_data(use_file, specific_panel):

            with open(use_file, "r") as g:
                lines = g.readlines()

            panel_size = (64, 64)
            data_names = [
                "x",
                "y",
                "frame",
                "tof",
                "panel",
                "hkl",
                "wavelength",
                "bbox",
            ]
            data = {i: [] for i in data_names}

            recording_table = False
            panel_num = -1
            tof_vals = self._get_time_channels_in_seconds()
            tof_vals = [i * 10**6 for i in tof_vals]
            tof_curve_coeffs = get_tof_curve_coefficients(tof_vals)

            for count, line in enumerate(lines):

                if recording_table:
                    if is_data_row(line) and is_active_row(count, lines):
                        if specific_panel is not None:
                            if specific_panel != panel_num:
                                continue
                        data["hkl"].append(get_hkl(line))
                        x = get_x(line)
                        y = get_y(line)
                        dx = get_dx(lines[count + 1])
                        dy = get_dy(lines[count + 1])

                        tof = get_tof(line)
                        wavelength = get_pixel_wavelength_in_ang(
                            x,
                            y,
                            tof * 10**-6,
                            8.3,
                            get_panel_l(panel_num),
                            pixel_size,
                            panel_size,
                        )
                        dtof = get_dtof(lines[count + 1])
                        x, y = convert_coord_to_dials(x, y)
                        bbox = get_bbox(x, dx, y, dy, tof, dtof, tof_curve_coeffs)

                        data["panel"].append(panel_num)
                        data["tof"].append(tof)
                        data["wavelength"].append(wavelength)
                        data["x"].append(x)
                        data["y"].append(y)
                        data["frame"].append(get_tof_frame(tof, tof_curve_coeffs))
                        data["bbox"].append(bbox)

                if at_start_of_table(line):
                    recording_table = True

                if at_end_of_table(line):
                    panel_num += 1
                    recording_table = False

            return data

        def calc_s0(unit_s0, wavelength):
            return unit_s0 * (1.0 / wavelength)

        pixel_size = (3, 3)
        data = get_data(use_file, specific_panel)
        nrows = len(data["hkl"])

        unit_s0 = np.array(self._get_unit_s0())
        use_reflection_table = dials_array_family_flex_ext.reflection_table(nrows)
        use_reflection_table["xyzobs.px.value"] = cctbx.array_family.flex.vec3_double(
            nrows
        )
        use_reflection_table["xyzobs.mm.value"] = cctbx.array_family.flex.vec3_double(
            nrows
        )
        use_reflection_table["tof_wavelength"] = cctbx.array_family.flex.double(nrows)
        use_reflection_table["tof"] = cctbx.array_family.flex.double(nrows)
        use_reflection_table["tof_s0"] = cctbx.array_family.flex.vec3_double(nrows)
        use_reflection_table["bbox"] = dials_array_family_flex_ext.int6(nrows)
        use_reflection_table["miller_index"] = cctbx.array_family.flex.miller_index(
            nrows
        )
        use_reflection_table["panel"] = cctbx.array_family.flex.size_t(nrows)
        use_reflection_table["flags"] = cctbx.array_family.flex.size_t(nrows, 32)

        for i in range(nrows):
            use_reflection_table["xyzobs.px.value"][i] = (
                data["x"][i],
                data["y"][i],
                data["frame"][i],
            )
            use_reflection_table["xyzobs.mm.value"][i] = (
                data["x"][i] * pixel_size[0],
                data["y"][i] * pixel_size[1],
                data["tof"][i],
            )
            use_reflection_table["tof_wavelength"][i] = data["wavelength"][i]
            use_reflection_table["tof"][i] = data["tof"][i]
            use_reflection_table["tof_s0"][i] = calc_s0(unit_s0, data["wavelength"][i])
            use_reflection_table["miller_index"][i] = data["hkl"][i]
            use_reflection_table["panel"][i] = data["panel"][i]
            use_reflection_table["bbox"][i] = data["bbox"][i]

        use_reflection_table.set_flags(
            use_reflection_table["miller_index"] != (0, 0, 0),
            use_reflection_table.flags.indexed,
        )

        return use_reflection_table


if __name__ == "__main__":
    for arg in argv[1:]:
        print(FormatISISSXD.understand(arg))
