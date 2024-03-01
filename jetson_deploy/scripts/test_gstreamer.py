import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Initialize GStreamer
Gst.init(None)

# Create the empty pipeline
pipeline = Gst.Pipeline.new("video-capture-pipeline")

# Create elements
source = Gst.ElementFactory.make("v4l2src", "source")
source.set_property("device", "/dev/video0")
source.set_property("num-buffers", 300)

capsfilter = Gst.ElementFactory.make("capsfilter", "caps")
caps = Gst.Caps.from_string("video/x-raw,width=1920,height=1080,framerate=30/1")
capsfilter.set_property("caps", caps)

converter = Gst.ElementFactory.make("nvvidconv", "converter")

nvmm_capsfilter = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
nvmm_caps = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=I420")
nvmm_capsfilter.set_property("caps", nvmm_caps)

encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
encoder.set_property("bitrate", 8000000)

parser = Gst.ElementFactory.make("h264parse", "parser")

muxer = Gst.ElementFactory.make("qtmux", "muxer")

sink = Gst.ElementFactory.make("filesink", "sink")
sink.set_property("location", "output.mp4")

# Check for errors
if not pipeline or not source or not capsfilter or not converter or not nvmm_capsfilter or not encoder or not parser or not muxer or not sink:
    print("ERROR: Not all elements could be created.")
    exit(-1)

# Build the pipeline
pipeline.add(source)
pipeline.add(capsfilter)
pipeline.add(converter)
pipeline.add(nvmm_capsfilter)
pipeline.add(encoder)
pipeline.add(parser)
pipeline.add(muxer)
pipeline.add(sink)


if not source.link(capsfilter):
    print("ERROR: Could not link source to capsfilter")
    exit(-1)

if not capsfilter.link(converter):
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


# Start playing
ret = pipeline.set_state(Gst.State.PLAYING)
if ret == Gst.StateChangeReturn.FAILURE:
    print("ERROR: Unable to set the pipeline to the playing state.")
    exit(-1)

# Wait until error or EOS
bus = pipeline.get_bus()
msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)

# Parse message
if msg:
    if msg.type == Gst.MessageType.ERROR:
        err, debug = msg.parse_error()
        print("ERROR:", err, debug)
    elif msg.type == Gst.MessageType.EOS:
        print("End of Stream")

pipeline.set_state(Gst.State.NULL)
