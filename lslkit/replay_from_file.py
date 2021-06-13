import time
from lslkit.components import outlets


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    import argparse
    description = "LSL Kit CLI interface"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--file-path', type=str, help='path to hdf, mat or csv file')
    parser.add_argument('--data-key', type=str)
    parser.add_argument('--stream-type', type=str, help='Data modality such as EEG, ECoG, EMG, etc.')
    parser.add_argument('--sample-rate', type=int, default=1)
    parser.add_argument('--chunksize', type=int, default=1)

    options = parser.parse_args()
    f_out = outlets.FileReplayOutlet(file_path=options.file_path, stream_type=options.stream_type,
                                     srate=options.sample_rate, data_key=options.data_key,
                                     outlet_chunksize=options.chunksize)

    f_out.begin()
    print("Launched")
    while(True):
        time.sleep(1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
