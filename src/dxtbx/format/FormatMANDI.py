from __future__ import annotations

from os.path import join
from sys import argv

import h5py
import numpy as np

from dials.array_family import flex

import dxtbx_flumpy as flumpy
from dxtbx import IncorrectFormatError
from dxtbx.format.FormatNXTOFRAW import FormatNXTOFRAW
from dxtbx.model import Detector
from dxtbx.model.beam import PolyBeamFactory
from dxtbx.model.sequence import SequenceFactory


class FormatMANDI(FormatNXTOFRAW):

    """
    Class to read NXTOFRAW from MaNDi
    (https://neutrons.ornl.gov/mandi)
    """

    def __init__(self, image_file: str, **kwargs: dict) -> None:
        super().__init__(image_file)
        if not FormatMANDI.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        self.image_file = image_file
        self.nxs_file = self.open_file(image_file)
        self._base_entry = self.get_base_entry_name(self.nxs_file)
        self.detector = None
        self.raw_data = None

    def open_file(self, image_file_path: str) -> h5py.File:
        return h5py.File(image_file_path, "r")

    @staticmethod
    def understand(image_file):
        try:
            return FormatMANDI.is_mandi_file(image_file)
        except IOError:
            return False

    @staticmethod
    def is_mandi_file(image_file):

        """
        Confirms if image_file is a NXTOFRAW format
        and from the SXD by confirming required fields
        are present and then checking the name attribute

        """

        def get_name(image_file):
            with h5py.File(image_file, "r") as handle:
                base_entry = list(handle.keys())[0]
                return handle[f"{base_entry}/instrument/name"][0].decode()

        if not FormatNXTOFRAW.understand(image_file):
            return False

        return get_name(image_file) == "MANDI"

    def load_raw_data(self, as_numpy_arrays=False, normalise_by_proton_charge=True):

        raw_data = []
        panel_size = self._get_panel_size_in_px()
        for panel_name in self._get_panel_names():
            spectra = np.reshape(
                self._nxs_file[self._base_entry][panel_name]["spectra"], panel_size
            )
            if as_numpy_arrays:
                raw_data.append(spectra)
            else:
                raw_data.append(flumpy.as_flex(spectra))

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

    def _get_time_channel_bins(self):
        return self.nxs_file[self._base_entry]["time_of_flight"][:]

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

    def _get_sample_to_moderator_distance(self):
        return 30000

    def _get_num_panels(self):
        return 40

    def _get_panel_names(self):
        return (
            "bank1",
            "bank2",
            "bank3",
            "bank5",
            "bank7",
            "bank8",
            "bank11",
            "bank12",
            "bank13",
            "bank17",
            "bank18",
            "bank19",
            "bank20",
            "bank21",
            "bank22",
            "bank23",
            "bank26",
            "bank27",
            "bank28",
            "bank29",
            "bank31",
            "bank32",
            "bank33",
            "bank37",
            "bank39",
            "bank40",
            "bank41",
            "bank42",
            "bank43",
            "bank46",
            "bank47",
            "bank48",
            "bank49",
            "bank50",
            "bank51",
            "bank52",
            "bank53",
            "bank57",
            "bank58",
            "bank59",
        )

    def _get_panel_gain(self):
        return 1.0

    def _get_panel_trusted_range(self):
        return (-1, 100000)

    def _get_panel_origins(self):
        return (
            (109.29204286444644, 2.0765839577139245, -421.88884104474835),
            (59.75559472749409, 196.3009651704246, -321.0886148477112),
            (-370.2739115575438, 222.21019427157452, -274.3289694122815),
            (-48.36730226344401, -47.655598074346614, -34.38823412842897),
            (262.58071883262494, -213.29901941660532, -438.2121401638587),
            (-101.44722535710339, -105.02142885050813, -293.8697118420855),
            (-339.86677764459404, -12.433617229374, -415.61553737760573),
            (-229.5136559344643, -30.283324678997698, 477.79552168604556),
            (-148.08735236588723, 86.78024874556917, 204.58676171945035),
            (-56.37546199899052, -161.3237680555105, 165.46683736675158),
            (10.151865053528677, -84.29652780207233, 474.844625871153),
            (209.83009991408272, -127.78491362390243, -410.97612240197583),
            (-54.980633655736824, -109.51388174845977, 244.33146337267738),
            (-426.71869511229454, 162.90240004187234, 391.27984645636735),
            (-392.69598531864676, 167.95142816127338, -301.27440318622587),
            (-346.1116632002346, 262.30274552635746, 308.99866845600417),
            (178.64667964939872, -422.309128614763, 292.0476460742577),
            (227.57698931254117, -352.21275385983995, -273.1750084322102),
            (257.0431241688841, -329.4216173044808, 374.57902356005525),
            (-95.54626629159752, -33.47663276915759, 243.64669891830266),
            (-92.78043199342954, 239.20951131695003, 102.25216022557728),
            (264.64751336680877, -23.240310204946926, -333.38140809120426),
            (-47.94425592066693, -377.7514032688696, 40.951433853672135),
            (-49.29046613774924, 34.12912877609427, 72.87132269638711),
            (-43.91175579129943, -168.86578379304953, 102.25781333004971),
            (-227.00318600679745, 251.1147187384595, 297.22758196347445),
            (-137.01865355179942, 0.7670976054394013, 424.38514940155585),
            (-226.43924768336433, 84.9048267058751, -252.96282352684264),
            (-72.13462050613772, -15.371409411072984, 324.1578376065953),
            (155.17563397475698, -385.35205076349365, 300.7204681135377),
            (118.60093333335669, -219.96889414944815, -221.85995684188185),
            (229.02227328744837, -267.279541415419, 417.6798749329017),
            (-165.48916221338962, 187.6280310422787, 275.3235605056206),
            (256.933250717166, 210.5499881344714, 410.04102168011815),
            (-446.6802347487758, 4.547693589407626, -355.0223179384615),
            (355.7172635281888, -143.65888978312333, 538.2981176608048),
            (103.41710904285672, -422.5012464333078, 223.08171053819936),
            (69.10664053962776, -433.77103904788083, 181.72828134073674),
            (386.74461649231074, -154.6817872125792, 536.4132084764059),
            (-85.1583756943751, -43.79828479164345, -353.2661947124004),
        )

    def _get_panel_slow_axes(self):
        return (
            (-0.40995944189101724, -0.0, 0.9121037528726683),
            (0.8413140916400035, -0.3993417417632542, 0.36430313269782144),
            (0.9135486805116172, -8.701305094112867e-08, 0.40672940431627935),
            (0.9136712554760633, -0.005691347651604999, 0.4064141305106196),
            (0.9000877040174273, -3.3423624969956044e-05, 0.43570876048054996),
            (0.8237654389174449, -0.4339318929151448, 0.3648473844722319),
            (0.5242060822793687, 0.753564207050922, 0.39667237003984906),
            (0.558631809841218, 0.7239243882647048, 0.40480116243545805),
            (0.5302677550126444, 0.7495756299966202, 0.3961722894259768),
            (0.5105858228713629, 0.7625393873654153, 0.3972855398817715),
            (0.5234993662324707, 0.7613043860394914, 0.38257554201913474),
            (0.5242805895802373, 0.76193525468196, 0.3802427265078654),
            (0.2773752175224878, 0.9263606175141056, 0.2547920623238439),
            (0.2824369424499928, 0.9253578256627166, 0.25286808423418894),
            (0.29341953582778163, 0.9189328008786695, 0.2635668481882594),
            (0.2756430556390824, 0.9267002116863475, 0.2554361437589211),
            (0.27564493927277767, 0.9266980734731193, 0.255441868288102),
            (0.26419452594778814, 0.9326825560789692, 0.24556974985780294),
            (0.2821833143009244, 0.9251972934047642, 0.2537371581117844),
            (0.25708833740035003, 0.9373331799382293, 0.23518523882168052),
            (0.005847357805351551, 0.9999666600196926, 0.00569975927182025),
            (-0.013532840864933613, 0.9998176850026921, -0.013470670880917773),
            (0.02073559994005865, 0.9995636405186222, 0.02103719202470605),
            (-0.01611539392584848, 0.9997372487982604, -0.01630114854542105),
            (-0.0015597814103797494, 0.99999763686379, -0.00151438066145792),
            (-0.26985366777513314, 0.9318454257086171, -0.24257596866597142),
            (-0.2689613295965925, 0.930586442992356, -0.2483317847164712),
            (-0.24478974228619782, 0.9409538500679607, -0.23384574854749585),
            (-0.27564482069205054, 0.9266981389574968, -0.25544175868162045),
            (-0.2756429368692951, 0.926700277112741, -0.2554360345629386),
            (-0.29616409168872754, 0.9158155450472476, -0.2712355401565291),
            (-0.2895114346035727, 0.921520535691634, -0.2588108025805443),
            (-0.28762806039436806, 0.9207015010343748, -0.26377802195562683),
            (-0.529079741353842, 0.7494032960942244, -0.398082060751378),
            (-0.5266154245579311, 0.7567985016981681, -0.38721082428707726),
            (-0.5183136573434037, 0.7608039423000866, -0.3905487344646644),
            (-0.5450814153066774, 0.7328386976902257, -0.4072268334171552),
            (-0.5335912913109352, 0.7456878489948737, -0.39903629621693226),
            (-0.5463489177242368, 0.7426823651882105, -0.38720248519347167),
            (-0.5364056285039712, 0.7488220847502627, -0.3892743082962578),
        )

    def _get_panel_fast_axes(self):
        return (
            (-0.9120924404554286, -0.004988374132461949, -0.4099542610921065),
            (-0.3973629504608723, 0.0, 0.9176615310674358),
            (0.4067294840933906, 0.0, -0.9135486449933164),
            (-0.40642079259222635, 0.0, 0.9136860179233928),
            (-0.4357088392373134, 0.0, 0.9000876665139196),
            (-0.4049605495302804, 0.0, 0.9143341584585656),
            (0.6034194048343725, 0.0, -0.7974239912802545),
            (0.5867710446352369, 0.0, -0.8097528889591398),
            (0.5985211351690538, 0.0, -0.8011070157949856),
            (0.6140978818836074, 0.0, -0.7892298723857751),
            (0.5900350712672735, 0.0, -0.8073776159113054),
            (0.5871086138418562, 0.0, -0.8095081689227688),
            (0.6764910739514576, 0.0, -0.7364508312603114),
            (0.6670312671894015, 0.0, -0.7450297232941122),
            (0.6682490613664352, 0.0, -0.7439376264061914),
            (0.6797104656750506, 0.0, -0.7334805265661836),
            (0.6797161629649926, 0.0, -0.7334752468925912),
            (0.6808176985558116, 0.0, -0.7324529072460345),
            (0.6686330451765915, 0.0, -0.7435925301520168),
            (0.6749772170482913, 0.0, -0.7378385707360002),
            (0.6980113712306184, 0.0, -0.716086674664982),
            (0.7054769991982411, 0.0, -0.7087328153840803),
            (0.712193476396768, 0.0, -0.7019832278465677),
            (0.7111470711844715, 0.0, -0.7030432725983148),
            (0.6965880529762186, 0.0, -0.7174713126326381),
            (0.6685195687395001, 0.0, -0.7436945516893025),
            (0.678368344951331, 0.0, -0.7347219804579089),
            (0.6907583140513702, 0.0, -0.7230857152294661),
            (0.6797161629888229, 0.0, -0.7334752468705076),
            (0.6797104655824727, 0.0, -0.7334805266519746),
            (0.6753887742946104, 0.0, -0.7374618658322775),
            (0.6664720065837039, 0.0, -0.7455300560274493),
            (0.6758903219427207, 0.0, -0.7370022202844204),
            (0.6012291238722038, 0.0, -0.7990766800550884),
            (0.5923841664428661, 0.0, -0.8056556332253818),
            (0.6017869642349561, 0.0, -0.7986566531851317),
            (0.5985088722933968, 0.0, -0.8011161774587294),
            (0.5988879863579233, 0.0, -0.8008328038961391),
            (0.5782215801374058, 0.0, -0.8158797731672245),
            (0.5873436180610783, 0.0, -0.8093376763273301),
        )

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
        wavelength_range = (0.2, 10)
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
            wavelengths=tof_in_seconds,
        )

    def get_goniometer(self, idx=None):
        return None

    def _get_panel_size_in_px(self):
        return (256, 256)

    def _get_panel_pixel_size_in_mm(self):
        return (0.621, 0.618)

    def _get_panel_size_in_mm(self):
        size_in_px = self._get_panel_size_in_px()
        pixel_size_in_mm = self._get_panel_pixel_size_in_mm()
        return tuple(
            [size_in_px[i] * pixel_size_in_mm[i] for i in range(len(size_in_px))]
        )

    def _get_panel_type(self) -> str:
        return "SENSOR_PAD"

    def get_pixel_spectra(self, panel_idx: int, x: int, y: int):
        raise NotImplementedError

    def _get_panel_projections_2d(self) -> dict:
        p_w, p_h = self._get_panel_size_in_px()
        panel_pos = {}
        count = 1
        for i in range(8):
            for j in range(5):
                panel_pos[count] = ((1, 0, 0, 1), (p_h * i, p_w * j))
                count += 1
        return panel_pos

    @staticmethod
    def write_histogram_data(
        nxs_file_path,
        tof_bins,
        panel_size,
        remove_event_data=True,
        spectra_output_name="spectra",
        write_tof_bins=True,
    ):

        """
        Generates histogram spectra for a given detector and writes it to nxs_file
        """

        def delete_event_data(nxs_file, base_dir, panel_name):
            del nxs_file[join(join(base_dir, panel_name), "event_index")]
            del nxs_file[join(join(base_dir, panel_name), "event_id")]
            del nxs_file[join(join(base_dir, panel_name), "event_time_zero")]
            del nxs_file[join(join(base_dir, panel_name), "event_time_offset")]

        nxs_file = h5py.File(nxs_file_path, "r+")
        base_dir = list(nxs_file.keys())[0]

        panel_names = FormatMANDI.get_panel_names(nxs_file)
        written_tof_bins = False
        for panel_name in panel_names:
            print(panel_name)
            output_path = join(base_dir, panel_name)
            output_path = join(output_path, spectra_output_name)
            panel_spectra, bin_edges = FormatMANDI.generate_histogram_data_for_panel(
                nxs_file, tof_bins, panel_size, panel_name
            )
            nxs_file.create_dataset(output_path, data=panel_spectra, compression="gzip")
            if remove_event_data:
                delete_event_data(nxs_file, base_dir, panel_name)
            if write_tof_bins and not written_tof_bins:
                nxs_file.create_dataset(
                    join(base_dir, "time_of_flight"), data=bin_edges, compression="gzip"
                )
                written_tof_bins = True

        nxs_file.close()

    @staticmethod
    def generate_histogram_data_for_panel(nxs_file, tof_bins, panel_size, panel_name):

        """
        Generates histogram data for a given panel
        """

        ## Get panel data
        panel_number = FormatMANDI.get_panel_number(panel_name)
        # Event id array idxs that were triggered after every pulse
        event_index = nxs_file[f"entry/{panel_name}/event_index"][:]
        # Actual pixel ids, starting from bottom left and going up y axis first
        event_id = nxs_file[f"entry/{panel_name}/event_id"][:]
        # Time of each pulse (usec)
        event_time_zero = nxs_file[f"entry/{panel_name}/event_time_zero"][:]
        # Time each event_id was triggered after event_time_zero (usec)
        event_time_offset = nxs_file[f"entry/{panel_name}/event_time_offset"][:]

        # event_index range is one more than event_id, so truncate
        event_index = event_index[event_index < event_id.shape[0]]

        # event_ids are given with an offset
        event_id_offset = panel_number * panel_size[0] * panel_size[1] - 1
        corrected_event_id = event_id[event_index] - event_id_offset
        event_time = (
            event_time_zero[event_index] + event_time_offset[corrected_event_id]
        )

        # Generate histograms
        spectra = []
        for i in range(panel_size[0] * panel_size[1]):
            h, edges = np.histogram(event_time[corrected_event_id == i], tof_bins)
            spectra.append(h)

        return np.array(spectra, dtype=np.dtype("i4")), np.array(edges)

    @staticmethod
    def get_time_range_for_panel(nxs_file, panel_size, panel_name):

        """
        Returns the range of event times for a given panel
        """

        def event_data_is_valid(event_id, event_time_offset):
            if len(event_id) == 0 or len(event_time_offset) == 0:
                return False
            return len(event_id) == len(event_time_offset)

        panel_number = FormatMANDI.get_panel_number(panel_name)
        event_index = nxs_file[f"entry/{panel_name}/event_index"]
        event_id = nxs_file[f"entry/{panel_name}/event_id"]
        event_time_zero = nxs_file[f"entry/{panel_name}/event_time_zero"]
        event_time_offset = nxs_file[f"entry/{panel_name}/event_time_offset"]

        if not event_data_is_valid(event_id, event_time_offset):
            return None, None

        num_pixels = panel_size[0] * panel_size[1]
        event_id_offset = panel_number * num_pixels - 1

        raw_event_id = event_id[event_index[0]]
        corrected_event_id = raw_event_id - event_id_offset
        min_event_time = event_time_zero[0] + event_time_offset[corrected_event_id]

        max_idx = int(event_index[-1] - 1)
        raw_event_id = event_id[max_idx]
        corrected_event_id = raw_event_id - event_id_offset
        max_event_time = (
            event_time_zero[max_idx] + event_time_offset[corrected_event_id]
        )

        return min_event_time, max_event_time

    @staticmethod
    def get_time_range_for_dataset(nxs_file_path, panel_size):

        """
        Iterates over num_panels to find the overall min/max tof event recorded
        """

        # TODO get panel_size and panel_number from nxs_file xml

        nxs_file = h5py.File(nxs_file_path, "r")

        min_tof = -1
        max_tof = -1

        panel_names = FormatMANDI.get_panel_names(nxs_file)

        for panel_name in panel_names:
            try:
                min_event_time, max_event_time = FormatMANDI.get_time_range_for_panel(
                    nxs_file, panel_size, panel_name
                )
                if min_event_time is None or max_event_time is None:
                    # Some banks contain no data
                    continue
                if min_tof < 0 or min_event_time < min_tof:
                    min_tof = min_event_time
                if max_event_time > max_tof:
                    max_tof = max_event_time
            except KeyError:
                # Some banks not always present
                pass

        nxs_file.close()

        return min_tof, max_tof

    @staticmethod
    def get_tof_bins(nxs_file, panel_size, delta_tof=5, padding=100):

        """
        delta_tof: float (usec)
        padding: float (usec)
        """

        min_tof, max_tof = FormatMANDI.get_time_range_for_dataset(nxs_file, panel_size)
        min_tof = min_tof - padding
        max_tof = max_tof + padding
        num_bins = int((max_tof - min_tof) / delta_tof)
        return np.linspace(min_tof, max_tof, num_bins)

    @staticmethod
    def get_panel_names(nxs_file):
        raw_names = [i for i in nxs_file[list(nxs_file.keys())[0]] if "bank" in i]
        raw_names = [
            (i, int("".join([j for j in i if j.isdigit()]))) for i in raw_names
        ]
        return [i[0] for i in sorted(raw_names, key=lambda x: x[0])]

    @staticmethod
    def get_panel_number(panel_name):
        return int("".join([i for i in panel_name if i.isdigit()]))


if __name__ == "__main__":
    for arg in argv[1:]:
        print(FormatMANDI.understand(arg))
