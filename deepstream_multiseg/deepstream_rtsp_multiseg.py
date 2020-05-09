################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

#!/usr/bin/env python

import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GstRtspServer
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call

import pyds, math


def main(args):

    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("usage: %s <config> <encoder>\n" % args[0])
        sys.exit(1)

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Set the model output resolution
    num_srcs=2
    if args[1]=='prisma_config.txt':
       op_size=256
    if args[1]=='deeplab_config.txt':
       op_size=513

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    

    # Source elements for reading from camera1
    print("Creating Source: Cam1... \n ")
     
    source_cam1 = Gst.ElementFactory.make("v4l2src", "camera-source1")
    source_cam1.set_property("device", "/dev/video1")
    vidconv_src1 = Gst.ElementFactory.make("videoconvert", "vidconv_src1")
    nvvidconv_src1 = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv_src1")
    filter_src1 = Gst.ElementFactory.make ("capsfilter", "filter_src1")
    nvvidconv_src1.set_property("nvbuf-memory-type", 0)
    caps_filter_src1 =Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12, width=1280, height=720, framerate=20/1") # Set max webcam resolution
    filter_src1.set_property("caps", caps_filter_src1)
    
    if not source_cam1:
        sys.stderr.write(" Unable to create source: cam1 \n")

    
    # Source elements for reading from camera2
    print("Creating Source: Cam2... \n ") 

    source_cam2 = Gst.ElementFactory.make("v4l2src", "camera-source2")
    source_cam2.set_property("device", "/dev/video0")
    vidconv_src2 = Gst.ElementFactory.make("videoconvert", "vidconv_src2")
    nvvidconv_src2 = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv_src2")
    filter_src2 =Gst.ElementFactory.make ("capsfilter", "filter_src2")
    nvvidconv_src2.set_property("nvbuf-memory-type", 0)
    caps_filter_src2 =Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12, width=640, height=480, framerate=20/1") # Set max webcam resolution
    filter_src2.set_property("caps", caps_filter_src2)

    if not source_cam2:
        sys.stderr.write(" Unable to create source: cam2 \n")

    
    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)

    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    seg = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not seg:
        sys.stderr.write(" Unable to create seg \n")
    seg.set_property('config-file-path', args[1])
    seg.set_property('batch-size', 1)

    # Use nvsegvisual to visualizes segmentation results
    nvsegvisual =Gst.ElementFactory.make("nvsegvisual", "nvsegvisual")
    if not nvsegvisual:
        sys.stderr.write(" Unable to create nvsegvisual \n")
    nvsegvisual.set_property('batch-size', 1)
    nvsegvisual.set_property('width', op_size)
    nvsegvisual.set_property('height', op_size)
    
    # Use nvtiler to composite the batched frames into a 2D tiled array
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    tiler_rows=int(math.sqrt(num_srcs))
    tiler_columns=int(math.ceil((1.0*num_srcs)//tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", op_size*2)
    tiler.set_property("height", op_size)

    # Convert the output video
    nvvidconv_posttile = Gst.ElementFactory.make("nvvideoconvert", "convertor_posttile")
    if not nvvidconv_posttile:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")

   # Create a caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
    
    # Make the encoder
    codec=args[2]
    if codec == "H264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        print("Creating H264 Encoder")
    elif codec == "H265":
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        print("Creating H265 Encoder")
    
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property('bitrate', 4000000)
    if is_aarch64():
        encoder.set_property('preset-level', 1)
        encoder.set_property('insert-sps-pps', 1)
        encoder.set_property('bufapi-version', 1)
    
    # Make the payload-encode video into RTP packets
    if codec == "H264":
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        print("Creating H264 rtppay")
    elif codec == "H265":
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
        print("Creating H265 rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")
    
    print("Creating UDP Sink... \n")
    
    # Make the UDP sink
    updsink_port_num = 5400
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        sys.stderr.write(" Unable to create udpsink")
    
    sink.set_property('host', '224.224.255.255')
    sink.set_property('port', updsink_port_num)
    sink.set_property('sync', 0) 
    
    # Set up the pipeline
    print("Adding elements to Pipeline...\n")
 
    # Add all elements into the pipeline
    pipeline.add(source_cam1)
    pipeline.add(vidconv_src1)
    pipeline.add(nvvidconv_src1)
    pipeline.add(filter_src1)
    
    pipeline.add(source_cam2)
    pipeline.add(vidconv_src2)
    pipeline.add(nvvidconv_src2)
    pipeline.add(filter_src2)
 

    pipeline.add(streammux)
    pipeline.add(seg)
    pipeline.add(nvsegvisual)
    pipeline.add(tiler)
    pipeline.add(nvvidconv_posttile)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)

    # Connect the pipeline elements together
    print("Linking elements in the Pipeline...\n")

    # Link the elements together for first camera source
    # camera-source -> videoconvert -> nvvideoconvert -> 
    # capsfilter -> nvstreammux
    source_cam1.link(vidconv_src1)
    vidconv_src1.link(nvvidconv_src1)
    nvvidconv_src1.link(filter_src1)

    sinkpad1 = streammux.get_request_pad("sink_0")
    if not sinkpad1:
        sys.stderr.write(" Unable to get the sink pad of streammux for src1\n")
    srcpad1 = filter_src1.get_static_pad("src")
    if not srcpad1:
        sys.stderr.write(" Unable to get source pad of decoder for src1\n")
    srcpad1.link(sinkpad1)

   
    # Link the elements together for second camera source
    # camera-source -> videoconvert -> nvvideoconvert -> 
    # capsfilter -> nvstreammux
    source_cam2.link(vidconv_src2)
    vidconv_src2.link(nvvidconv_src2)
    nvvidconv_src2.link(filter_src2)

    sinkpad2 = streammux.get_request_pad("sink_1")
    if not sinkpad2:
        sys.stderr.write(" Unable to get the sink pad of streammux for src2\n")
    srcpad2 = filter_src2.get_static_pad("src")
    if not srcpad2:
        sys.stderr.write(" Unable to get source pad of decoder for src2\n")
    srcpad2.link(sinkpad2)


    #Link the elements together for encoding and streaming
    # nvstreammux -> nvinfer -> nvsegvisual -> nvtiler -> 
    # nvvideoconvert -> capsfilter -> nvv4l2h264enc -> rtph264pay -> udpsink
    streammux.link(seg)
    seg.link(nvsegvisual)
    nvsegvisual.link(tiler)
    tiler.link(nvvidconv_posttile)
    nvvidconv_posttile.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    
    # Start streaming
    rtsp_port_num = 8554
    
    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)
    
    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch( "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (updsink_port_num, codec))
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-seg", factory)
    
    print("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-seg ***\n\n" % rtsp_port_num)
    
    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

'''
Run application from directory: <DeepStream 4.0 ROOT>/sources/python
Sample run:-
Server Jetson: Run python3 deepstream_rtsp_multiseg.py prisma_config.txt H264
Client VLC: In another system open the rtsp stream: 'rtsp://<jetson-ip>:8554/ds-seg'
'''
