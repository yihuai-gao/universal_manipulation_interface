import time
import gi
import numpy as np
gi.require_version("Gst", '1.0')
from gi.repository import Gst, GLib # type: ignore
import numpy.typing as npt
import cv2
import sys
import os
def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t==Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True

class GstreamerPipeline:

    def __init__(self, resolution: tuple[int, int] = (1920, 1080), framerate: int=30, bitrate:int = 8000000):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("gst-video-record")

        self.appsrc = Gst.ElementFactory.make("appsrc", "video_source")
        appsrc_caps = Gst.Caps.from_string(f"video/x-raw,format=RGBA,width={resolution[0]},height={resolution[1]},framerate={framerate}/1")
        self.appsrc.set_property("caps", appsrc_caps)

        converter = Gst.ElementFactory.make("nvvidconv", "converter")

        nvmm_capsfilter = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
        nvmm_caps = Gst.Caps.from_string(f"video/x-raw(memory:NVMM),width={resolution[0]},height={resolution[1]},framerate={framerate}/1,format=I420")
        nvmm_capsfilter.set_property("caps", nvmm_caps)

        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        encoder.set_property("bitrate", bitrate)

        parser = Gst.ElementFactory.make("h264parse", "parser")

        muxer = Gst.ElementFactory.make("qtmux", "muxer")

        self.sink = Gst.ElementFactory.make("filesink", "sink")

        # Check for errors
        elements = [self.appsrc, converter, nvmm_capsfilter, encoder, parser, muxer, self.sink]
        for element in elements:
            if not element:
                print(f"Element could not be created. Exiting.")
                exit(-1)

        # Add elements to the pipeline
        for element in elements:
            self.pipeline.add(element)

        if not self.appsrc.link(converter):
            print("ERROR: Could not link appsrc to converter")
            exit(-1)

        if not converter.link(nvmm_capsfilter):
            print("ERROR: Could not link converter to nvmm_capsfilter")
            exit(-1)

        if not nvmm_capsfilter.link(encoder):
            print("ERROR: Could not link nvmm_capsfilter to encoder")
            exit(-1)

        if not encoder.link(parser):
            print("ERROR: Could not link encoder to parser")
            exit(-1)

        if not parser.link(muxer):
            print("ERROR: Could not link parser to muxer")
            exit(-1)

        if not muxer.link(self.sink):
            print("ERROR: Could not link muxer to sink")
            exit(-1)

        self.is_recording = False
        self.frame_cnt = 0
        self.frame_rate = framerate
        self.resolution = resolution
        self.output_file = ""

        

    # ========= context manager ===========
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_recording:
            self.appsrc.emit("end-of-stream")
            self.pipeline.set_state(Gst.State.NULL)
            self.is_recording = False

    def start_recording(self, output_file: str):
        start_time = time.monotonic()
        self.sink.set_property("location", output_file)
        # self.sink.set_property("location", output_file)
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("ERROR: Unable to set the pipeline to the playing state.")
            exit(-1)
        self.is_recording = True
        self.frame_cnt = 0
        self.output_file = output_file
        print(f"Recording started. Output file: {output_file}, starting time spent: {time.monotonic() - start_time} s")

    def stop_recording(self):
        if not self.is_recording:
            print("Pipeline is not in recording state. Command is ignored.")
            return
        self.appsrc.emit("end-of-stream")
        time.sleep(0.1)
        self.pipeline.set_state(Gst.State.NULL)
        self.is_recording = False
        print(f"Recording stopped. {self.frame_cnt} frames recorded")

    def encode_img(self, img: npt.NDArray[np.uint8]):
        assert img.shape == (self.resolution[1], self.resolution[0], 3), "Image should be of shape (height, width, 3) and in BGR format"
        assert self.is_recording, "Pipeline must be in recording state to encode frames"
        assert img.dtype == np.uint8, "Image must be of type np.uint8"
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        duration = 10**9 / self.frame_rate
        buf = Gst.Buffer.new_wrapped(frame.tobytes())
        self.frame_cnt += 1
        buf.pts = self.frame_cnt * 10**9/30
        buf.duration = 10**9/30
        self.appsrc.emit("push-buffer", buf)


if __name__ == "__main__":
    with GstreamerPipeline(bitrate=6000000) as pipeline:
        device_path = "/dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB320216MTR-video-index0"
        cap = cv2.VideoCapture(device_path)
        print(f"Capturing from device {device_path}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

        for total_duration_s in [10]:
            pipeline.start_recording(f"test_{total_duration_s}.mp4")
            start_time = time.monotonic()
            while time.monotonic() - start_time < total_duration_s:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                pipeline.encode_img(frame)
            pipeline.stop_recording()
        cap.release()


