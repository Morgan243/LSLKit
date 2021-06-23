from uuid import uuid4
import numpy as np
import pandas as pd
import pylsl
import attr
import os
from scipy.io import loadmat
from tqdm.auto import tqdm
import threading


@attr.s
class PeriodicCallback:
    delta_t_sec = attr.ib()
    callback = attr.ib()
    args = attr.ib(attr.Factory(lambda: list()))
    kwargs = attr.ib(attr.Factory(lambda: dict()))
    done = attr.ib(False, init=False)

    add_error = attr.ib(0, init=False)

    _start_t = attr.ib(None, init=False)
    _next_t = attr.ib(None, init=False)
    last_error = attr.ib(0, init=False)
    sum_error = attr.ib(0, init=False)
    done_event = attr.ib(None, init=False)


    def _period(self):
        t = pylsl.local_clock()

        if self._next_t is not None:
            this_error = self._next_t - t
            # This one works well - first iteration
            error = (this_error) * 0.25 + (self.last_error * 0.25) + (self.sum_error * .75)

            #error = (self._next_t - t) * 0. + (self.last_error * 0.25) + (self.sum_error * .75)
            #error = (self._next_t - t) * 1.0 + (self.last_error * 0.) + (self.sum_error * .75)
            #error = this_error * 1.0 + ((this_error - self.last_error) * .5) + (self.sum_error * .75)
            #error = this_error * .25 + ((this_error - self.last_error) * .25) + (self.sum_error * .75)
        else:
            error = 0

        self.last_error = error
        self.sum_error = (self.sum_error / 2. + error / 2.)
        self._next_t = t + self.delta_t_sec

        r = self.callback(*self.args, **self.kwargs)
        if not self.done:
            threading.Timer(self._next_t - pylsl.local_clock() + self.add_error + error, self._period).start()
        else:
            self.done_event.set()
        return r

    def set_additional_error(self, error):
        self.add_error = error
        return self

    def begin(self):
        self.done = False
        self.done_event = threading.Event()
        self._period()
        return self

    def end(self):
        self.done = True
        self.done_event.wait()
        return self


@attr.s
class BaseOutlet:
    name = attr.ib(None)
    stream_type = attr.ib(None)
    n_channels = attr.ib(None)

    srate = attr.ib(1)
    pbar = attr.ib(True)

    n_samples = attr.ib(None)
    buffer_time = attr.ib(0.)
    drop_error_rate = attr.ib(0.)
    channel_format = attr.ib('float32')
    outlet_chunksize = attr.ib(16)
    max_buffered = attr.ib(256)

    p_callback = attr.ib(None, init=False)
    _pbar = attr.ib(None, init=False)
    outlet_info = attr.ib(None, init=False)
    lsl_outlet = attr.ib(None, init=False)


    def __attrs_post_init__(self):
        self.sample_delta_t = 1. / self.srate
        self.init_lsl()

    def init_pbar(self):
        self._pbar = tqdm(desc=self.name, total=self.n_samples)

    def init_lsl(self):
        self.outlet_info = pylsl.StreamInfo(self.name, self.stream_type, channel_count=self.n_channels,
                                            nominal_srate=self.srate, channel_format=self.channel_format,
                                            source_id=self.name + self.stream_type + str(uuid4())[-5:])
        self.lsl_outlet = pylsl.StreamOutlet(self.outlet_info, chunk_size=self.outlet_chunksize,
                                             max_buffered=self.max_buffered)

    def increment(self, n=1) -> list:
        raise NotImplementedError()

    def push_to_lsl(self):
        out = self.increment(self.outlet_chunksize)
        stamp = pylsl.local_clock() - self.buffer_time
        if isinstance(out, np.ndarray):
            out = out.tolist()
        self.lsl_outlet.push_chunk(out, stamp)

    def begin(self):
        if self.pbar:
            self.init_pbar()

        assert self.p_callback is None, 'Callback handler already exists'
        self.p_callback = PeriodicCallback(self.sample_delta_t, self.push_to_lsl)
        self.p_callback.begin()
        return self

    def stop(self):
        self.p_callback.end()
        self._pbar.close()
        return self


@attr.s
class FileReplayOutlet(BaseOutlet):
    file_path = attr.ib(None)
    data_key = attr.ib(None)
    outlet_chunksize = attr.ib(32)

    on_end = attr.ib('restart')

    outlet_info = attr.ib(None, init=False)
    lsl_outlet = attr.ib(None, init=False)
    sent_samples = attr.ib(0, init=False)

    def __attrs_post_init__(self):
        assert self.file_path is not None, f'File path cannot be None'
        self.name = self.file_path if self.name is None else self.name
        self.df = self.load_file_to_frame(self.file_path, self.data_key)
        self.arr = self.df.values
        self.n_channels = self.df.shape[-1]
        self.n_samples = self.df.shape[0]
        super().__attrs_post_init__()

    @classmethod
    def load_file_to_frame(cls, p, key=None):
        fname = os.path.split(p)[-1]
        if '.csv' in fname.lower():
            df = pd.read_csv(p)
        elif '.mat' in fname.lower():
            mat_data = loadmat(p, variable_names=[key])
            df = pd.DataFrame(mat_data[key])
        elif '.hdf' in fname.lower():
            df = pd.read_hdf(p, key)
        else:
            raise ValueError("Don't know how to load " % fname)

        return df

    def increment(self, n=1):
        if (self.sent_samples + n) > self.n_samples:
            if self.on_end == 'raise':
                raise StopIteration()
            elif self.on_end == 'restart':
                print("Restarting")
                self.sent_samples = 0
                #self._pbar.close()
                self.init_pbar()

        s = self.arr[self.sent_samples: self.sent_samples + n]
        self.sent_samples += n
        self._pbar.update(n)
        return s


from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import logging
@attr.s
class BrainflowOutlet(BaseOutlet):

    serial_port = attr.ib('/dev/ttyACM0')
    board_id = attr.ib(1)
    timeout = attr.ib(0)
    streamer_params = attr.ib('')
    select_channels = attr.ib('all')
    buffer_size = attr.ib(128 * 250)

    lsl_outlet = attr.ib(None, init=False)

#    options = [
#        dict(dest='--serial-port', type=str, default=None, help='Serial port of device'),
#        dict(dest='--board-id', type=int, default=None, hell='brainflow board id'),
#        dict(dest='--timeout', type=int, default=0, help='Device discovery timeout')
#    ]

    def __attrs_post_init__(self):
        self.board_shim = self.build_brainflow_shim(self.serial_port, self.board_id, self.timeout)

        try:
            self.board_shim.prepare_session()
            self.board_shim.start_stream(self.buffer_size, self.streamer_params)
            self.board_desc_map = self.board_shim.get_board_descr(self.board_id)
            """
            Example board_desc_map (Ganglion):
            {'accel_channels': [5, 6, 7],
             'analog_channels': None,
             'ecg_channels': [1, 2, 3, 4],
             'eeg_channels': [1, 2, 3, 4],
             'emg_channels': [1, 2, 3, 4],
             'eog_channels': [1, 2, 3, 4],
             'marker_channel': 14,
             'name': 'Ganglion',
             'num_rows': 15,
             'package_num_channel': 0,
             'resistance_channels': [8, 9, 10, 11, 12],
             'sampling_rate': 200,
             'temperature_channels': None,
             'timestamp_channel': 13}
            """
        except BaseException as e:
            logging.warning('Exception', exc_info=True)
            raise

        types = [k.split('_')[0] for k, v in self.board_desc_map.items() if 'channel' in k and k != 'package_num_channel']
        type_str = "_".join(types)

        self.name = self.board_desc_map['name']
        self.n_channels = self.board_desc_map['num_rows']
        self.srate = self.board_desc_map['sampling_rate'] #/ 4
        self.stream_type = type_str
        print("Stream type: " + self.stream_type)
        super().__attrs_post_init__()

        #self.lsl_outlet = BaseOutlet(name=self.board_desc_map['name'], stream_type=type_str,
        #                             # TODO: is numb rows correct for getting all channels (what's from board_data
        #                             n_channels=self.board_desc_map['num_rows'])

    @staticmethod
    def build_brainflow_shim(serial_port: str, board_id: int, timeout=0, mac_address='', other_info='',
                             serial_number=''):#, streamer_params='', buffer_size=188 * 250):
        BoardShim.enable_dev_board_logger()

        params = BrainFlowInputParams()
        params.ip_port = 0
        params.serial_port = serial_port
        params.mac_address = mac_address
        params.other_info = other_info
        params.serial_number = serial_number
        params.ip_address = ''
        params.ip_protocol = 0
        params.timeout = timeout
        params.file = ''

        board_shim = BoardShim(board_id, params)
        return board_shim

    def increment(self, n=None) -> list:
        # Brainflow data is channel per row - transpose for lsl and dataframes
        # Reshape to ensure always 2d
        data = self.board_shim.get_board_data().T.reshape(-1, self.n_channels)
        if self._pbar is not None:
            self._pbar.update(data.shape[0])
            #self._pbar.set_description()
        #print("Data: " + str(data))
        #print("-----")
        return data

    def end(self):
        logging.info('End')
        if self.board_shim and self.board_shim.is_prepared():
            logging.info('Releasing session')
            self.board_shim.release_session()
