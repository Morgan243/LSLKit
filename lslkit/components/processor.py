import pylsl
from pylsl import resolve_stream, resolve_bypred, resolve_byprop, StreamInlet
import attr
import numpy as np
import pandas as pd

@attr.s
class ProcessStream:
    stream_info = attr.ib()
    process_f = attr.ib()
    chunksize: int = attr.ib()
    preprocessor = attr.ib(None)
    pbar = attr.ib(True)
    #max_buf = attr.ib(None)
    #stream_type = attr.ib('EMG')

    done = attr.ib(False, init=False)
    #resolve_timeout = 10
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __attrs_post_init__(self):
        import pylsl
        #self.streams = resolve_byprop('type', 'EMG', timeout=10)
        #self.streams = self.find_streams(self.stream_type, self.resolve_timeout)
        self.buffer = np.empty((self.chunksize, self.stream_info.channel_count()),
                               dtype=self.dtypes[self.stream_info.channel_format()])

        self.stream_creation_t = self.stream_info.created_at()
        self.stream_creation_timesamp = pd.Timestamp.fromtimestamp(self.stream_creation_t)

        if self.pbar:
            self.init_pbar()

        self.init_lsl()

    def init_lsl(self):
        self.inlet = StreamInlet(self.stream_info, max_buflen=self.chunksize, processing_flags=pylsl.proc_ALL)
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
    def from_resolve(cls, process_f, stream_type, preprocessor=None, chunksize=None, timeout=10, filter_f=None):
        streams = cls.find_streams(stream_type=stream_type, timeout=timeout, filter_f=filter_f)
        s = None if len(streams) == 0 else streams[0]
        if len(streams) == 0:
            raise ValueError("No Streams found :(")
        elif len(streams) > 1:
            print(f"{len(streams)} available, selecting the first")

        #print(f"-----Selected stream ----\n{print((s.as_xml()))}\n---------")
        return cls(stream_info=s, process_f=process_f, preprocessor=preprocessor, chunksize=chunksize)

    def increment(self):
        _, timestamps = self.inlet.pull_chunk(timeout=0, max_samples=self.buffer.shape[0], dest_obj=self.buffer)
        #chunk_arr = np.array(chunk)
        if timestamps is not None and len(timestamps) > 0:
            ts = pd.Series(map(pd.Timestamp.fromtimestamp, timestamps))
            chunk_df = pd.DataFrame(self.buffer[:ts.size, :], index=ts - self.stream_creation_timesamp)
        else:
            chunk_df = None
        return chunk_df

    def begin(self):
        while not self.done:
            chunk_df = self.increment()
            if chunk_df is not None:
                self.process_f(chunk_df)
                self._pbar.update(n=chunk_df.shape[0])