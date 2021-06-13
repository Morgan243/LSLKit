from uuid import uuid4
import numpy as np
import pandas as pd
import pylsl
import attr
import os
from scipy.io import loadmat
from tqdm.auto import tqdm
import threading
import time


#class Counter():
#    def __init__(self, increment):
#        self.next_t = time.time()
#        self.i = 0
#        self.done = False
#        self.increment = increment
#        self._run()
#
#    def _run(self):
#        print("hello ", self.i)
#        self.next_t += self.increment
#        self.i += 1
#        if not self.done:
#            threading.Timer(self.next_t - time.time(), self._run).start()
#
#    def stop(self):
#        self.done = True
#
#
#a = Counter(increment=1)
#time.sleep(5)
#a.stop()

@attr.s
class PeriodicCallback:
    delta_t_sec = attr.ib()
    callback = attr.ib()
    args = attr.ib(attr.Factory(lambda: list()))
    kwargs = attr.ib(attr.Factory(lambda: dict()))
    done = attr.ib(False, init=False)

    #start_time = attr.ib(0, init=False)
    next_time = attr.ib(None, init=False)
    add_error = attr.ib(0, init=False)

    def _period(self):
        t = pylsl.local_clock()
#        if self.next_time is not None:
#            # If we are late, it's negative
#            error = 0#(self.next_time - t)
#            #print(error)
#        else:
#            error = 0
        #self.start_time = t
        self.next_time = t + self.delta_t_sec
        self.callback(*self.args, **self.kwargs)
        if not self.done:
            threading.Timer(self.next_time - pylsl.local_clock() + self.add_error, self._period).start()

    def set_additional_error(self, error):
        self.add_error = error
        return self

    def begin(self):
        self._period()
        return self

    def end(self):
        self.done = True
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
    outlet_chunksize = attr.ib(512)
    max_buffered = attr.ib(512)

    p_callback = attr.ib(None, init=False)
    _pbar = attr.ib(None, init=False)
    _start_t = attr.ib(None, init=False)
    _next_t = attr.ib(None, init=False)
    last_error = attr.ib(0, init=False)
    sum_error = attr.ib(0, init=False)

    def __attrs_post_init__(self):
        self.sample_delta_t = 1. / self.srate
        if self.pbar:
            self.init_pbar()
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
        self._next_t = t + self.sample_delta_t
        out = self.increment(self.outlet_chunksize)
        stamp = t - self.buffer_time
        if isinstance(out, np.ndarray):
            out = out.tolist()
        self.lsl_outlet.push_chunk(out, stamp)
        self.p_callback.set_additional_error(error)

    def begin(self):
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
    #outlet_chunksize = attr.ib(32)
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
        #self.sample_delta_t = 1. / self.srate
        #if self.pbar:
        #    self.init_pbar()
        #self.init_lsl()

    @classmethod
    def load_file_to_frame(cls, p, key=None):
        fname = os.path.split(p)[-1]
        if '.csv' in fname.lower():
            df = pd.read_csv(p)
        elif '.mat' in fname.lower():
            mat_data = loadmat(p)
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
                self.n_samples = 0
                self._pbar.close()
                self.init_pbar()

        s = self.arr[self.sent_samples: self.sent_samples + n]
        self.sent_samples += n
        self._pbar.update(n)
        return s




#    def run(self):
#        print("now sending data...")
#        start_time = pylsl.local_clock()
#        sent_samples = 0
#        #
#        sleep_time = self.sample_delta_t * 0.5
#        print("Start time: " + str(start_time))
#        while True:
#            elapsed_time = pylsl.local_clock() - start_time
#            required_samples = int(self.srate * elapsed_time) - sent_samples
#            if required_samples > 0:
#                print("Sending required samples: " + str(required_samples))
#                # make a chunk==array of length required_samples, where each element in the array
#                # is a new random n_channels sample vector
#                #mychunk = [[rand() for chan_ix in range(self.n_channels)]
#                #           for samp_ix in range(required_samples)]
#                mychunk = self.arr[sent_samples: sent_samples + required_samples].tolist()
#                #mychunk.tolist()
#                # get a time stamp in seconds (we pretend that our samples are actually
#                # 125ms old, e.g., as if coming from some external hardware)
#                stamp = pylsl.local_clock() - self.buffer_time
#                # now send it and wait for a bit
#                # Note that even though `rand()` returns a 64-bit value, the `push_chunk` method
#                #  will convert it to c_float before passing the data to liblsl.
#                self.lsl_outlet.push_chunk(mychunk, stamp)
#                sent_samples += required_samples
#            time.sleep(sleep_time)

