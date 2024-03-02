import sys
import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from jetson_deploy.modules.video_recorder_jetson import VideoRecorderJetson
import cv2
import json
import time
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.uvc_camera import UvcCamera, VideoRecorder
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths


def test():
    # Find and reset all Elgato capture cards.
    # Required to workaround a firmware bug.
    reset_all_elgato_devices()
    time.sleep(0.1)
    v4l_paths = get_sorted_v4l_paths()
    v4l_path = v4l_paths[0]
    
    with SharedMemoryManager() as shm_manager:
        # video_recorder = VideoRecorder.create_h264(
        #     shm_manager=shm_manager,
        #     fps=30,
        #     codec='h264_nvenc',
        #     input_pix_fmt='bgr24',
        #     thread_type='FRAME',
        #     thread_count=4
        # )
        video_recorder = VideoRecorderJetson(
            shm_manager=shm_manager,
            fps=30,
            codec='h264_nvenc',
            input_pix_fmt='bgr24',
            bit_rate=6000*1000
        )

        with UvcCamera(
            shm_manager=shm_manager,
            dev_video_path=v4l_path,
            resolution=(1920, 1080),
            capture_fps=30,
            video_recorder=video_recorder,
            put_downsample=False,
            verbose=True
        ) as camera:
            cv2.setNumThreads(1) 
            
            rec_start_time = time.time() + 2
            # camera.start_recording(video_path, start_time=rec_start_time)

            data = None
            episode_cnt = 0
            while True:
                data = camera.get(out=data)
                t = time.time()
                # print('capture_latency', data['receive_timestamp']-data['capture_timestamp'], 'receive_latency', t - data['receive_timestamp'])
                # print('receive', t - data['receive_timestamp'])

                dt = time.time() - data['timestamp']
                print(data['camera_capture_timestamp'] - data['camera_receive_timestamp'])

                bgr = data['color']
                cv2.imshow('default', bgr)
                key = cv2.pollKey()
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    video_path = f"test_{episode_cnt}.mp4"
                    episode_cnt += 1
                    camera.start_recording(video_path, time.time())
                    print("Recording started")
                elif key == ord('s'):
                    camera.stop_recording()
                    print("Recording stopped")
                
                time.sleep(1/60)


if __name__ == "__main__":
    test()
