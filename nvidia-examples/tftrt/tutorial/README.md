
# mixed Tensorflow-TensorRT model conversion example

This example shows how to convert a frozen tensorflow graphdef into a mixed
Tensorflow-TensorRT model for fast inference execution in native Tensorflow:
 * Graph is traversed and TensorRT compatible subgraph is identified
 * Calibration op is inserted before each TRT subgraph (for int8 conversion)
 * TRT subgraph would be converted to a TensorRT engine and wrapped in custom op
 * return a mixed TF-TRT graphdef that could be loaded and executed

## python API
To enable code, we should import the contrib model in your python script.
```
from tensorflow.contrib import tensorrt as trt
```

### memory allocation
Currently TensorRT engine uses independent memory allocation outside of TF
When starting tensorflow session, remember to use
 * allow_growth 
 * per_process_gpu_memory_fraction
to restrict Tensorflow from claiming all GPU memory

```
  config=tf.ConfigProto(gpu_options=
             tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
             allow_growth=True))
```

### FP32/FP16 conversion

FP32/FP16 inference graph could be easily converted through a single call to
function trt.create_inference_graph.
```
  trt.create_inference_graph(
      input_graph_def=orig_graph,       # native Tensorflow graphdef
      outputs=["output"],               # list of names for output node
      max_batch_size=batch_size,        # maximum/optimum batchsize for TF-TRT
                                        # mixed graphdef
      max_workspace_size_bytes=1 << 25, # maximum workspace for each
                                        # TRT engine to allocate
      precision_mode=precision,         # TRT Engine precision
                                        # "FP32","FP16" or "INT8"
      minimum_segment_size=2            # minimum number of nodes in an engine,
                                        # this parameter allows the converter to
                                        # skip subgraph with total node number
                                        # less than the threshold
```

**max_batch_size** is also the target optimal batch size for inference.
Performance drop is expected if following session run provide input with batch
size smaller than max_batch_size.

**outputs** a **list** of output node names.

**max_workspace_size_bytes** is the maximum memory each TensorRT engine could
allocate. Specified in bytes, (1<<30 = 1GB)

**precision_mode** specifies target precision ("FP32", "FP16" or "INT8") 

**minimum_segment_size** (optional) threshold for subgraph size to trigger
conversion. Any subgraph with node # < **minimum_segment_size** would be left
unchanged.

Function returns a frozen graph ready to be imported & executed.

### INT8 conversion

INT8 conversion should start calling the same function as FP32/FP16, it returns
a calibration graph with calibration op inserted before each subgraph.
```
  int8_calib_graph= trt.create_inference_graph(graph_def, ["outputs"],
                        batch_size, precision_mode="INT8")
```

To feed calibration data, create a session with the calibration graph.
```
  g = tf.Graph()
  with g.as_default():
    inp, out = tf.import_graph_def(
        graph_def=int8_calib_gdef, return_elements=["input", "output"])
    inp = inp.outputs[0]
    out = out.outputs[0]
```

Start TF session with TF-TRT graph, execute the graph and feed it with input
calibration_batch should be sharded and feed through TF-TRT mixed network

Calibration data should be representatitive of the inference dataset to reduce
quantization error.

Start session with restricted GPU memory allocation.
To feed more data into calibration network, do iterative runs while each
iteration we feed the **max_batch_size** into the network
```
  with tf.Session(graph=g, config=config) as sess:
    iteration = int(CALIBRATION_BATCH/batch_size)
    # iterate through the clibration data, each time we feed data with
    #   batch size < BATCH_SIZE (specified during conversion)
    for i in range(iteration):
      val = sess.run(out, {inp: dummy_input[i::iteration]})
```

Once finished feeding calibration, trigger calib_graph_to_infer_graph.
Calibration process might take some time, depending on how many calibration data
has been fed to the model.
```
  #   TF-TRT mixed graphdef for inference
  int8_graph = trt.calib_graph_to_infer_graph(int8_calib_gdef)
  return int8_graph
```

clib_graph_to_infer_graph returns a frozen graph.

### execution

Import mixed TF-TRT graphdef.
```
  g = tf.Graph()
  with g.as_default():
    inp, out = tf.import_graph_def(
        graph_def=graph_def, return_elements=["input", "output"])
    inp = inp.outputs[0]
    out = out.outputs[0]
```

Start Tensorflow session with restricted GPU memory allocation.
```
  with tf.Session(graph=g, config=config) as sess:
```

Call session run to trigger TensorRT subgraph execution. 
```
    val = sess.run(out, {inp: dummy_input})
```
