from lslkit.components.outlets import BrainflowOutlet
import time

if __name__ == """__main__""":
    import argparse

    description = "LSL Kit CLI interface"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--serial-port', default='/dev/ttyACM0', type=str, help='path to hdf, mat or csv file')
    parser.add_argument('--board-id', default=1, type=int, help="brainflow board id")
    parser.add_argument('--streamer-params', type=str)
    parser.add_argument('--buffer-size', type=int, default=1024*10)
    parser.add_argument('--timeout', type=int, default=1024*10)
    #parser.add_argument('--chunksize', type=int, default=1)

    options = parser.parse_args()
    bfo = BrainflowOutlet(options.serial_port, options.board_id, options.timeout)

    #f_out = outlets.FileReplayOutlet(file_path=options.file_path, stream_type=options.stream_type,
    #                                 srate=options.sample_rate, data_key=options.data_key,
    #                                 outlet_chunksize=options.chunksize)

    bfo.begin()
    print("Launched")
    while (True):
        time.sleep(1)
