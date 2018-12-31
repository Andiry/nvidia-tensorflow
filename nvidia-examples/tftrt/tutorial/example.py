# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Script to test TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib import tensorrt as trt

################################################################################
# construct a simple graphdef
################################################################################
def get_simple_graph_def():
  """Create a simple graph and return its graph_def."""
  g = tf.Graph()
  with g.as_default():
    a = tf.placeholder(
        dtype=tf.float32, shape=(None, 24, 24, 2), name="input")
    e = tf.constant(
        [[[[1., 0.5, 4., 6., 0.5, 1.], [1., 0.5, 1., 1., 0.5, 1.]]]],
        name="weights",
        dtype=tf.float32)
    conv = tf.nn.conv2d(
        input=a, filter=e, strides=[1, 2, 2, 1], padding="SAME", name="conv")
    b = tf.constant(
        [4., 1.5, 2., 3., 5., 7.], name="bias", dtype=tf.float32)
    t = tf.nn.bias_add(conv, b, name="biasAdd")
    relu = tf.nn.relu(t, "relu")
    idty = tf.identity(relu, "ID")
    v = tf.nn.max_pool(
        idty, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")
    tf.squeeze(v, name="output")
  return g.as_graph_def()
################################################################################


################################################################################
# execute a graphdef
################################################################################
def run_graphdef(graph_def, input_data):
  # load TF-TRT graph into memory and extract input & output nodes
  g = tf.Graph()
  with g.as_default():
    inp, out = tf.import_graph_def(
        graph_def=graph_def, return_elements=["input", "output"])
    inp = inp.outputs[0]
    out = out.outputs[0]
  # allow_growth and restrict Tensorflow to claim all GPU memory
  # currently TensorRT engine uses independent memory allocation outside of TF
  config=tf.ConfigProto(gpu_options=
             tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
             allow_growth=True))
  # we can now import trt_graph into Tensorflow and execute it. If given target
  with tf.Session(graph=g, config=config) as sess:
    val = sess.run(out, {inp: dummy_input})
  return val
################################################################################


################################################################################
# conversion example
################################################################################
def convert_tftrt_fp(orig_graph, batch_size, precision):
  # convert native Tensorflow graphdef into a mixed TF-TRT graph
  trt_graph = trt.create_inference_graph(
      input_graph_def=orig_graph,       # native Tensorflow graphdef
      outputs=["output"],               # list of names for output node
      max_batch_size=batch_size,        # maximum/optimum batchsize for TF-TRT
                                        # mixed graphdef
      max_workspace_size_bytes=1 << 25, # maximum workspace (in MB) for each 
                                        # TRT engine to allocate
      precision_mode=precision,         # TRT Engine precision
                                        # "FP32","FP16" or "INT8"
      minimum_segment_size=2            # minimum number of nodes in an engine,
                                        # this parameter allows the converter to
                                        # skip subgraph with total node number
                                        # less than the threshold
  )
  
  # allow_growth and restrict Tensorflow to claim all GPU memory
  # currently TensorRT engine uses independent memory allocation outside of TF
  config=tf.ConfigProto(gpu_options=
             tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
             allow_growth=True))
  # we can now import trt_graph into Tensorflow and execute it. If given target
  # precision_mode as 'FP32' or 'FP16'.
  if precision=='FP16' or precision=='FP32':
    return trt_graph

  # 'INT8' precision would require an extra step of calibration
  int8_calib_gdef = trt_graph
  # 'INT8' precision requires calibration to retrieve proper quantization range.
  # trt.create_inference_graph returns a calibration graph def with inserted
  #   calibration op that captures input tensor during session run to feed
  #   TensorRT subgraph during engine construction

  # feed calibration date into TF-TRT mixed graph
  # this step is just running the calibration graph with a set of representative
  #   input data. (could use a subset of validation data with even distribution
  #   of all categories)
  g = tf.Graph()
  with g.as_default():
    inp, out = tf.import_graph_def(
        graph_def=int8_calib_gdef, return_elements=["input", "output"])
    inp = inp.outputs[0]
    out = out.outputs[0]
  
  # start TF session with TF-TRT graph, execute the graph and feed it with input
  #   calibration_batch should be sharded and feed through TF-TRT mixed network
  # Should use real data that is representatitive of the inference dataset for
  #   calibration to reduce quantization error. 
  # For this test script it is random data.
  CALIBRATION_BATCH=100
  inp_dims = (CALIBRATION_BATCH, 24, 24, 2)
  dummy_input = np.random.random_sample(inp_dims)
  
  # allow_growth and restrict Tensorflow to claim all GPU memory
  # currently TensorRT engine uses independent memory allocation outside of TF
  config=tf.ConfigProto(gpu_options=
             tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
             allow_growth=True))
  
  # start session to feed calibration data
  with tf.Session(graph=g, config=config) as sess:
    iteration = int(CALIBRATION_BATCH/batch_size)
    # iterate through the clibration data, each time we feed data with
    #   batch size < BATCH_SIZE (specified during conversion)
    for i in range(iteration):
      val = sess.run(out, {inp: dummy_input[i::iteration]})
  
  # finished calibration, trigger calib_graph_to_infer_graph to build
  #   TF-TRT mixed graphdef for inference
  int8_graph = trt.calib_graph_to_infer_graph(int8_calib_gdef)
  return int8_graph
################################################################################

# The optima(also maximum) batch size for converted TF-TRT mixed model
BATCH_SIZE=5

# generate a trivial graph and return the frozen graphdef
orig_graph = get_simple_graph_def()

# start TF session with TF-TRT graph, execute the graph and feed it with input
inp_dims = (BATCH_SIZE, 24, 24, 2)
dummy_input = np.random.random_sample(inp_dims)

# tf 2 trt conversion for FP32
tftrt_fp32 = convert_tftrt_fp(orig_graph, BATCH_SIZE, "FP32")
# tf 2 trt conversion for FP16
tftrt_fp16 = convert_tftrt_fp(orig_graph, BATCH_SIZE, "FP16")
# tf 2 trt conversion for int8
tftrt_int8 = convert_tftrt_fp(orig_graph, BATCH_SIZE, "INT8")

# execute each graph
tf_res = run_graphdef(orig_graph, dummy_input)
tftrt_fp32_res = run_graphdef(tftrt_fp32, dummy_input)
tftrt_fp16_res = run_graphdef(tftrt_fp16, dummy_input)
tftrt_int8_res = run_graphdef(tftrt_int8, dummy_input)
