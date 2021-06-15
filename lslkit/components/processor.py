import pylsl
from pylsl import resolve_stream, resolve_bypred, resolve_byprop, StreamInlet
import attr
import numpy as np
import pandas as pd

@attr.s
class ProcessStream:
    stream_info = attr.ib()
    process_f = attr.ib()
    max_buflen: int = attr.ib()
    preprocessor = attr.ib(None)
    pbar = attr.ib(True)
    #max_buf = attr.ib(None)
    #stream_type = attr.ib('EMG')

    done = attr.ib(False, init=False)
    _pbar = attr.ib(None, init=False)
    #resolve_timeout = 10
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __attrs_post_init__(self):
        import pylsl
        #self.streams = resolve_byprop('type', 'EMG', timeout=10)
        #self.streams = self.find_streams(self.stream_type, self.resolve_timeout)
        self.buffer = np.empty((self.max_buflen, self.stream_info.channel_count()),
                               dtype=self.dtypes[self.stream_info.channel_format()])

        self.stream_creation_t = self.stream_info.created_at()
        self.stream_creation_timesamp = pd.Timestamp.fromtimestamp(self.stream_creation_t)

        if self.pbar:
            self.init_pbar()

        self.init_lsl()

    def init_lsl(self):
        self.inlet = StreamInlet(self.stream_info, max_buflen=self.max_buflen, processing_flags=pylsl.proc_ALL)
        return self

    def init_pbar(self):
        from tqdm.auto import tqdm
        self._pbar = tqdm(desc=str(self.process_f))
        return self

    @classmethod
    def find_streams(cls, stream_type, timeout=10, filter_f=None):
        streams = resolve_byprop('type', stream_type, timeout=timeout)
        print(f"Found {len(streams)} {stream_type} streams")
        if filter_f is not None:
            print(f"...filtering with {str(filter_f)}")
            streams = [s for s in streams if filter_f(s)]
            print(f"{len(streams)} remain after filtering")
        return streams

    @classmethod
    def from_resolve(cls, process_f, stream_type, preprocessor=None, max_buflen=None, timeout=10, filter_f=None):
        streams = cls.find_streams(stream_type=stream_type, timeout=timeout, filter_f=filter_f)
        s = None if len(streams) == 0 else streams[0]
        if len(streams) == 0:
            raise ValueError("No Streams found :(")
        elif len(streams) > 1:
            print(f"{len(streams)} available, selecting the first")

        #print(f"-----Selected stream ----\n{print((s.as_xml()))}\n---------")
        return cls(stream_info=s, process_f=process_f, preprocessor=preprocessor, max_buflen=max_buflen)


    def increment(self):
        _, timestamps = self.inlet.pull_chunk(timeout=0, max_samples=self.buffer.shape[0], dest_obj=self.buffer)
        #chunk_arr = np.array(chunk)
        if timestamps is not None and len(timestamps) > 0:
            ts = pd.Series(map(pd.Timestamp.fromtimestamp, timestamps))
            chunk_df = pd.DataFrame(self.buffer[:ts.size, :], index=ts - self.stream_creation_timesamp)
        else:
            chunk_df = None
        return chunk_df

    def increment_until(self, chunk_size):
        buffer = np.empty((chunk_size, self.stream_info.channel_count()), dtype=self.buffer.dtype)
        # Pull as much data as we could use (full buffer)
        _, timestamps = self.inlet.pull_chunk(timeout=0, max_samples=buffer.shape[0], dest_obj=buffer)
        # Until chunk_size is met, load data into buffer
        while len(timestamps) < chunk_size:
            remaining = chunk_size - len(timestamps)
            #b = buffer[buffer.shape[0] - remaining:]
            b = buffer[len(timestamps):]
            _, new_timestamps = self.inlet.pull_chunk(timeout=0, max_samples=remaining, dest_obj=b)
            timestamps += new_timestamps

        ts = pd.Series(map(pd.Timestamp.fromtimestamp, timestamps))
        chunk_df = pd.DataFrame(buffer, index=ts - self.stream_creation_timesamp)
        return chunk_df

    def increment_and_process(self, required_size=None):
        chunk_df = self.increment() if required_size is None else self.increment_until(required_size)
        if chunk_df is not None:
            r = self.process_f(chunk_df)
            self._pbar.update(n=chunk_df.shape[0])
        else:
            r = None
        return chunk_df, r

    def begin(self, required_size=None):
        while not self.done:
            in_data, proc_out = self.increment_and_process(required_size=required_size)
            if self._pbar is not None:
                self._pbar.update(in_data.shape[0])
                self._pbar.set_description(f"{self.inlet.samples_available()} samples available")

    def apply(self, other_as_in):
        def new_proc_func(_x):
            __x = self.process_f(_x)
            return other_as_in(__x)
        self.process_f = new_proc_func
        return self
