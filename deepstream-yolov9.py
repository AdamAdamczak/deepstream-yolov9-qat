#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import time
import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')

from gi.repository import GLib, Gst
import logging
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
import argparse
import configparser
from cffi import FFI

logging.basicConfig(level=logging.INFO)
ffi = FFI()

# Definitions of C structures and functions
ffi.cdef("""
typedef struct {
  uint32_t source_id;
  uint32_t frame_num;
  double comp_in_timestamp;
  double latency;
} NvDsFrameLatencyInfo;

uint32_t nvds_measure_buffer_latency(void *buf, NvDsFrameLatencyInfo *latency_info);
bool nvds_get_enable_latency_measurement();
""")

# Load the NVIDIA Deepstream metadata library
clib = ffi.dlopen("/opt/nvidia/deepstream/deepstream/lib/libnvdsgst_meta.so")

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_BATCH_TIMEOUT_USEC = 33000
total_latency = 0
total_frames = 0
last_timestamp = None

def generate_config_file(gpu_id, onnx_file, precision):
    """Generate a configuration file for the model."""  
    config = configparser.ConfigParser()
    config['property'] = {
        'gpu-id': str(gpu_id),
        'net-scale-factor': '0.0039215697906911373',
        'model-color-format': '0',
        'onnx-file': onnx_file,
        'model-engine-file': f'model_b1_gpu{gpu_id}_{precision}.engine',
        'int8-calib-file': 'calib.table' if precision == 'int8' else '',
        'labelfile-path': 'labels.txt',
        'batch-size': '1',
        'network-mode': '2' if precision == 'fp16' else '1' if precision == 'int8' else '0',
        'num-detected-classes': '601',
        'interval': '0',
        'gie-unique-id': '1',
        'process-mode': '1',
        'network-type': '0',
        'cluster-mode': '2',
        'maintain-aspect-ratio': '1',
        'symmetric-padding': '1',
        'workspace-size': '2000',
        'parse-bbox-func-name': 'NvDsInferParseYoloCuda',
        'custom-lib-path': 'nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so',
        'engine-create-func-name': 'NvDsInferYoloCudaEngineGet'
    }
    config['class-attrs-all'] = {
        'nms-iou-threshold': '0.45',
        'pre-cluster-threshold': '0.25',
        'topk': '300'
    }
    with open('dstest1_pgie_config.txt', 'w') as configfile:
        config.write(configfile)
def build_pipeline(media_file, gpu_id):
    """Build the GStreamer pipeline."""
    pipeline = Gst.Pipeline()
    if not pipeline:
        logging.error("build_pipeline: Unable to create Pipeline")
        return None

    logging.info("build_pipeline: Creating Source")
    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        logging.error("build_pipeline: Unable to create Source")
        return None
    source.set_property('location', media_file)

    logging.info("build_pipeline: Creating H264Parser")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parser:
        logging.error("build_pipeline: Unable to create H264 parser")
        return None

    logging.info("build_pipeline: Creating Decoder")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        logging.error("build_pipeline: Unable to create Nvv4l2 Decoder")
        return None

    logging.info("build_pipeline: Creating StreamMux")
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        logging.error("build_pipeline: Unable to create NvStreamMux")
        return None
    streammux.set_property('width', 640)
    streammux.set_property('height', 384)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)

    logging.info("build_pipeline: Creating Inference")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        logging.error("build_pipeline: Unable to create primary inference")
        return None
    pgie.set_property('config-file-path', "dstest1_pgie_config.txt")



    logging.info("build_pipeline: Creating Converter")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "converter")
    if not nvvidconv:
        logging.error("build_pipeline: Unable to create Converter")
        return None

    logging.info("build_pipeline: Creating OSD")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        logging.error("build_pipeline: Unable to create OSD")
        return None

    logging.info("build_pipeline: Creating Fake Output")
    fake_sink = Gst.ElementFactory.make("fakesink", "fake-output")
    if not fake_sink:
        logging.error("build_pipeline: Unable to create Fake Sink")
        return None

    logging.info("build_pipeline: Adding elements to Pipeline")
    elements = [source, h264parser, decoder, streammux, nvvidconv, pgie, nvosd, fake_sink]


    for element in elements:
        pipeline.add(element)

    logging.info("build_pipeline: Linking elements in the Pipeline")
    source.link(h264parser)
    h264parser.link(decoder)
    decoder_srcpad = decoder.get_static_pad("src")
    streammux_sinkpad = streammux.get_request_pad("sink_0")
    decoder_srcpad.link(streammux_sinkpad)
    streammux.link(pgie)
    pgie.link(nvosd)
    nvosd.link(nvvidconv)
    nvvidconv.link(fake_sink)

    if measure_latency:
        osd_src_pad = nvosd.get_static_pad("src")
        osd_src_pad.add_probe(Gst.PadProbeType.BUFFER, osd_src_pad_buffer_probe, 1)

    logging.info("build_pipeline: Pipeline construction complete")
    return pipeline


def osd_src_pad_buffer_probe(pad, info, u_data):
    """
    Buffer probe for OSD source pad to measure frame latency and log information.
    """
    global last_timestamp, total_latency,total_frames
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        logging.error("osd_src_pad_buffer_probe: Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    # Checking if latency measurement is enabled
    if clib.nvds_get_enable_latency_measurement():
        c_gst_buf = ffi.cast("void *", hash(gst_buffer))
        cNvDsFrameLatencyInfo = ffi.new("NvDsFrameLatencyInfo[]", u_data)
        source_count = clib.nvds_measure_buffer_latency(c_gst_buf, cNvDsFrameLatencyInfo)
        
        for i in range(source_count):
            logging.info(f"osd_src_pad_buffer_probe: Source id = {cNvDsFrameLatencyInfo[i].source_id}, "
                         f"Frame_num = {cNvDsFrameLatencyInfo[i].frame_num}, "
                         f"Frame latency = {cNvDsFrameLatencyInfo[i].latency} ms")
            if last_timestamp is not None:
                frame_latency = cNvDsFrameLatencyInfo[i].latency - last_timestamp
                logging.info(f"osd_src_pad_buffer_probe: Latency between frames: {frame_latency} ms")
                total_latency += frame_latency
            last_timestamp = cNvDsFrameLatencyInfo[i].latency
            total_frames += 1

    return Gst.PadProbeReturn.OK



        
        

def main(args):
    global measure_latency
    parser = argparse.ArgumentParser(description="Run the GStreamer pipeline with specified model and settings.")
    parser.add_argument("media_file", help="Path to the media file or URI.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--onnx-file", default="default.onnx", help="Path to the ONNX model file.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32", help="Model precision mode.")
    parser.add_argument("--measure-latency", action="store_true", help="Enable latency measurement.")
    args = parser.parse_args()  
    measure_latency = args.measure_latency
    
    generate_config_file(args.gpu_id, args.onnx_file, args.precision)
    Gst.init(None)
    pipeline = build_pipeline(args.media_file, args.gpu_id)
    if not pipeline:
        logging.error("Pipeline creation failed")
        sys.exit(1)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    logging.info("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        logging.info("Pipeline execution interrupted")
    finally:
        pipeline.set_state(Gst.State.NULL)
        logging.info("Pipeline stopped")

if __name__ == '__main__':
    sys.exit(main(sys.argv))