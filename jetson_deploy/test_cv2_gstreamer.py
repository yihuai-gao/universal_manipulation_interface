import time
import traceback
import sys
import cv2
import gi
import threading
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib  # noqa:F401,F402
import numpy as np
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

Gst.init(None)
# loop = GLib.MainLoop()
pipeline = Gst.Pipeline.new("video-capture")

# Create elements
appsrc = Gst.ElementFactory.make("appsrc", "video_source")
appsrc_caps = Gst.Caps.from_string("video/x-raw,format=RGBA,width=1920,height=1080,framerate=30/1")
appsrc.set_property("caps", appsrc_caps)
# appsrc.set_property("num-buffers", 100)


converter = Gst.ElementFactory.make("nvvidconv", "converter")

nvmm_capsfilter = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
nvmm_caps = Gst.Caps.from_string("video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1,format=I420")
nvmm_capsfilter.set_property("caps", nvmm_caps)

encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
encoder.set_property("bitrate", 8000000)

parser = Gst.ElementFactory.make("h264parse", "parser")

muxer = Gst.ElementFactory.make("qtmux", "muxer")

sink = Gst.ElementFactory.make("filesink", "sink")
sink.set_property("location", "output_cam.mp4")

# Check for errors
elements = [appsrc, converter, nvmm_capsfilter, encoder, parser, muxer, sink]
for element in elements:
    if not element:
        print(f"Element could not be created. Exiting.")
        exit(-1)

# Add elements to the pipeline
for element in elements:
    pipeline.add(element)

if not appsrc.link(converter):
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

if not muxer.link(sink):
    print("ERROR: Could not link muxer to sink")
    exit(-1)



# Function to capture and push frames
def capture_and_push_frames():
    cap = cv2.VideoCapture('/dev/video0')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    cnt = 0
    prev_time = time.monotonic()
    pts = 0  # buffers presentation timestamp
    duration = 10**9 / 30
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # Convert to grayscale if needed
            pts += duration
            gst_buffer = Gst.Buffer.new_wrapped(frame.tobytes())
            gst_buffer.pts = pts
            gst_buffer.duration = duration
            appsrc.emit("push-buffer", gst_buffer)
            print(f"Frame {cnt} pushed, fps: {1 / (time.monotonic() - prev_time)}")
            prev_time = time.monotonic()
            cnt += 1
    except KeyboardInterrupt:
        cap.release()
        appsrc.emit("end-of-stream")




# Start playing
ret = pipeline.set_state(Gst.State.PLAYING)
if ret == Gst.StateChangeReturn.FAILURE:
    print("ERROR: Unable to set the pipeline to the playing state.")
    exit(-1)

# Push frames in a loop (You might need to run this in a separate thread or adjust according to your application's architecture)

capture_and_push_frames()

time.sleep(0.1)
pipeline.set_state(Gst.State.NULL)
