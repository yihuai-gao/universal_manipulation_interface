import sys
import os



ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import numpy as np
from jetson_deploy.modules.video_recorder_jetson import VideoRecorderJetson
import cv2
import json
import time
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.uvc_camera import UvcCamera, VideoRecorder
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from umi.common.cv_util import draw_predefined_mask

from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer
from umi.real_world.multi_uvc_camera import MultiUvcCamera
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
def test_single():
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


def test_multiple():

    shm_manager = SharedMemoryManager()
    shm_manager.start()

    reset_all_elgato_devices()
    time.sleep(0.1)
    v4l_paths = get_sorted_v4l_paths()
    multi_cam_vis_resolution = (960, 960)
    rw, rh, col, row = optimal_row_cols(
        n_cameras=len(v4l_paths),
        in_wh_ratio=4/3,
        max_resolution=multi_cam_vis_resolution
    )
    obs_image_resolution = (224, 224)
    obs_float32 = True
    max_obs_buffer_size=60
    camera_obs_latency = 0.125
    resolution = list()
    capture_fps = list()
    cap_buffer_size = list()
    video_recorder = list()
    transform = list()
    vis_transform = list()
    for path in v4l_paths:
        res = (1920, 1080)
        fps = 30
        buf = 1
        bit_rate = 6000*1000
        is_mirror = None
        mirror_mask = np.ones((224,224,3),dtype=np.uint8)
        mirror_mask = draw_predefined_mask(
            mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
        is_mirror = (mirror_mask[...,0] == 0)
        fisheye_converter = None
        def tf(data, input_res=res):
            img = data['color']
            if fisheye_converter is None:
                f = get_image_transform(
                    input_res=input_res,
                    output_res=obs_image_resolution, 
                    # obs output rgb
                    bgr_to_rgb=True)
                img = np.ascontiguousarray(f(img))
                if is_mirror is not None:
                    img[is_mirror] = img[:,::-1,:][is_mirror]
                img = draw_predefined_mask(img, color=(0,0,0), 
                    mirror=False, gripper=True, finger=False, use_aa=True)
            else:
                img = fisheye_converter.forward(img)
                img = img[...,::-1]
            if obs_float32:
                img = img.astype(np.float32) / 255
            data['color'] = img
            return data
        transform.append(tf)
        resolution.append(res)
        capture_fps.append(fps)
        cap_buffer_size.append(buf)
        video_recorder.append(VideoRecorderJetson(
            shm_manager=shm_manager,
            fps=fps,
            codec='h264_nvenc',
            input_pix_fmt='bgr24',
            bit_rate=bit_rate
        ))
        def vis_tf(data, input_res=res):
            img = data['color']
            f = get_image_transform(
                input_res=input_res,
                output_res=(rw,rh),
                bgr_to_rgb=False
            )
            img = f(img)
            data['color'] = img
            return data
        vis_transform.append(vis_tf)

    camera = MultiUvcCamera(
        dev_video_paths=v4l_paths,
        shm_manager=shm_manager,
        resolution=resolution,
        capture_fps=capture_fps,
        # send every frame immediately after arrival
        # ignores put_fps
        put_downsample=False,
        get_max_k=max_obs_buffer_size,
        receive_latency=camera_obs_latency,
        cap_buffer_size=cap_buffer_size,
        transform=transform,
        vis_transform=vis_transform,
        video_recorder=video_recorder,
        verbose=False
    )
    print(f"MultiUvcCamera start waiting")
    camera.start(wait=True)

    multi_cam_vis = MultiCameraVisualizer(
        camera=camera,
        row=row,
        col=col,
        rgb_to_bgr=False
    )
    multi_cam_vis.start(wait=False)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    # test_single()
    test_multiple()
