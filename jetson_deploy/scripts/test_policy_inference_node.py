import time
import numpy as np
import click
import zmq
@click.command()
@click.option('--output', '-o', required=True, help='Path to output')
def main(output):
    iter_idx = 0
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://robodog-adapter1:8766")
    while True:
        obs_dict_np = np.load(f'{output}/obs/obs_dict_np_{iter_idx}.npy', allow_pickle=True).item()
        send_start_time = time.monotonic()
        socket.send_pyobj(obs_dict_np)
        recv_start_time = time.monotonic()
        action = socket.recv_pyobj()
        print(f'Send time: {recv_start_time - send_start_time:.3f} s, Recv time: {time.monotonic() - recv_start_time:.3f} s')
        iter_idx += 6

if __name__ == '__main__':
    main()