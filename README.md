# LSLKit
Python library to help write Python code that interacts with Lab Streaming layer

### Replay From File
As a library
```python
from lslkit.components import outlets
file_out = outlets.FileReplayOutlet('my_lab_time_series.mat', stream_type='EEG',
                                    srate=300, data_key='eeg')
file_out.begin()
```

From the Command line
```bash
python -m lslkit.replay_from_file --file-path=my_lab_time_series.mat --data-key=eeg --stream-type=EEG --sample-rate=300 --chunksize=1
```

### Processing a stream
```python
from lslkit.components import processor
proc_f = lambda _df: _df.mean()
# Find a matching stream and build a processor around it
proc = processor.ProcessStream.from_resolve(proc_f, 'emg', max_buflen=512)
# Pull data until each dataframe has the required_size samples
proc.begin(required_size=51)
```