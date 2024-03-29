#!/usr/bin/env python
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

from __future__ import print_function
from builtins import range
import nvutils
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import horovod.tensorflow as hvd
import os
import sys
import time
import argparse
import random
import numpy as np

def _stage(tensors):
    """Stages the given tensors in a StagingArea for asynchronous put/get.
    """
    stage_area = data_flow_ops.StagingArea(
        dtypes=[tensor.dtype       for tensor in tensors],
        shapes=[tensor.get_shape() for tensor in tensors])
    put_op      = stage_area.put(tensors)
    get_tensors = stage_area.get()
    tf.add_to_collection('STAGING_AREA_PUTS', put_op)
    return put_op, get_tensors

class _PrefillStagingAreasHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        # TODO: This assumes TF collections are ordered; is this safe?
        enqueue_ops = tf.get_collection('STAGING_AREA_PUTS')
        for i in range(len(enqueue_ops)):
            session.run(enqueue_ops[:i+1])

class _LogSessionRunHook(tf.train.SessionRunHook):
    def __init__(self, global_batch_size, num_records, display_every=10):
        self.global_batch_size = global_batch_size
        self.num_records = num_records
        self.display_every = display_every
    def after_create_session(self, session, coord):
        print('  Step Epoch Img/sec   Loss  LR')
        self.elapsed_secs = 0.
        self.count = 0
    def before_run(self, run_context):
        self.t0 = time.time()
        return tf.train.SessionRunArgs(
            fetches=['step_update:0', 'loss:0', 'total_loss:0',
                     'learning_rate:0'])
    def after_run(self, run_context, run_values):
        self.elapsed_secs += time.time() - self.t0
        self.count += 1
        global_step, loss, total_loss, lr = run_values.results
        print_step = global_step + 1 # One-based index for printing.
        if print_step == 1 or print_step % self.display_every == 0:
            dt = self.elapsed_secs / self.count
            img_per_sec = self.global_batch_size / dt
            epoch = print_step * self.global_batch_size / self.num_records
            print('%6i %5.1f %7.1f %6.3f %6.3f %7.5f' %
                  (print_step, epoch, img_per_sec, loss, total_loss, lr))
            self.elapsed_secs = 0.
            self.count = 0

def _cnn_model_function(features, labels, mode, params):
    model_func    = params['model']
    model_format  = params['format']
    model_dtype   = params['dtype']
    momentum      = params['momentum']
    learning_rate_init = params['learning_rate_init']
    learning_rate_power = params['learning_rate_power']
    decay_steps   = params['decay_steps']
    weight_decay  = params['weight_decay']
    loss_scale    = params['loss_scale']
    larc_eta      = params['larc_eta']
    larc_mode     = params['larc_mode']
    deterministic = params['deterministic']
    num_classes   = params['n_classes']
    use_dali      = params['use_dali']

    device        = '/gpu:0'
    labels = tf.reshape(labels, (-1,)) # Squash unnecessary unary dim
    inputs = features # TODO: Should be using feature columns?
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    if is_training and not use_dali:
        with tf.device('/cpu:0'):
            # Stage inputs on the host
            preload_op, (inputs, labels) = _stage([inputs, labels])
        with tf.device(device):
            # Stage inputs to the device
            gpucopy_op, (inputs, labels) = _stage([inputs, labels])
    with tf.device(device):
        inputs = tf.cast(inputs, model_dtype)
        if not use_dali:
            imagenet_mean = tf.constant([121, 115, 100], dtype=model_dtype)
            imagenet_std  = tf.constant([70, 68, 71], dtype=model_dtype)
            inputs = tf.subtract(inputs, imagenet_mean)
            inputs = tf.multiply(inputs, 1. / imagenet_std)
        if model_format == 'channels_first':
            inputs = tf.transpose(inputs, [0,3,1,2])
        with nvutils.fp32_trainable_vars(
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay)):
            top_layer = model_func(inputs, training=is_training)
            logits = tf.layers.dense(top_layer, num_classes)
        predicted_classes = tf.argmax(logits, axis=1, output_type=tf.int32)
        logits = tf.cast(logits, tf.float32)
        if mode == tf.estimator.ModeKeys.PREDICT:
            probabilities = tf.softmax(logits)
            predictions = {
                'class_ids': predicted_classes[:, None],
                'probabilities': probabilities,
                'logits': logits
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        loss = tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels)
        loss = tf.identity(loss, name='loss') # For access by logger (TODO: Better way to access it?)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([loss] + reg_losses, name='total_loss')
        with tf.device(None): # Allow fallback to CPU if no GPU support for these ops
            top1_accuracy = tf.metrics.accuracy(
                labels=labels, predictions=predicted_classes)
            top5_accuracy = tf.metrics.mean(tf.nn.in_top_k(
                predictions=logits, targets=labels, k=5))
            tf.summary.scalar('top1_accuracy', top1_accuracy[1])
            tf.summary.scalar('top5_accuracy', top5_accuracy[1])
        if mode == tf.estimator.ModeKeys.EVAL:
            metrics = {'top1_accuracy': top1_accuracy,
                       'top5_accuracy': top5_accuracy}
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)
        assert(mode == tf.estimator.ModeKeys.TRAIN)
        #batch_size = inputs.shape[0]
        batch_size = tf.shape(inputs)[0]
        learning_rate = tf.train.polynomial_decay(
            learning_rate_init, tf.train.get_global_step(),
            decay_steps=decay_steps, end_learning_rate=0.,
            power=learning_rate_power, cycle=False, name='learning_rate')
        opt = tf.train.MomentumOptimizer(
            learning_rate, momentum, use_nesterov=True)
        opt = hvd.DistributedOptimizer(opt)
        opt = nvutils.LarcOptimizer(opt, learning_rate, larc_eta, clip=larc_mode)
        opt = nvutils.LossScalingOptimizer(opt, scale=loss_scale)
        gate_gradients = (tf.train.Optimizer.GATE_OP if deterministic else
                          tf.train.Optimizer.GATE_NONE)
        train_op = opt.minimize(
            loss, global_step=tf.train.get_global_step(),
            gate_gradients=gate_gradients, name='step_update')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) or []
        if use_dali:
            train_op = tf.group(train_op, update_ops)
        else:
            train_op = tf.group(preload_op, gpucopy_op, train_op, update_ops)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def _get_num_records(filenames):
    def count_records(tf_record_filename):
        count = 0
        for _ in tf.python_io.tf_record_iterator(tf_record_filename):
            count += 1
        return count
    nfile = len(filenames)
    return (count_records(filenames[0])*(nfile-1) +
            count_records(filenames[-1]))

def train(infer_func, params):
    image_width = params['image_width']
    image_height = params['image_height']
    image_format = params['image_format']
    batch_size = params['batch_size']
    distort_color = params['distort_color']
    data_dir = params['data_dir']
    data_idx_dir = params['data_idx_dir']
    log_dir = params['log_dir']
    precision = params['precision']
    momentum = params['momentum']
    learning_rate_init = params['learning_rate_init']
    learning_rate_power = params['learning_rate_power']
    weight_decay = params['weight_decay']
    loss_scale = params['loss_scale']
    larc_eta = params['larc_eta']
    larc_mode = params['larc_mode']
    num_iter = params['num_iter']
    checkpoint_secs = params['checkpoint_secs']
    display_every = params['display_every']
    iter_unit = params['iter_unit']
    use_dali = params['use_dali']

    # Determinism is not fully supported by all TF ops.
    # Disabling until remaining wrinkles can be ironed out.
    deterministic = False
    if deterministic:
        tf.set_random_seed(2 * (1 + hvd.rank()))
        random.seed(3 * (1 + hvd.rank()))
        np.random.seed(2)

    log_dir  = None if log_dir  == "" else log_dir
    data_dir = None if data_dir == "" else data_dir
    data_idx_dir = None if data_idx_dir == "" else data_idx_dir

    global_batch_size = batch_size * hvd.size()
    if data_dir is not None:
        filename_pattern = os.path.join(data_dir, '%s-*')
        train_filenames = sorted(tf.gfile.Glob(filename_pattern % 'train'))
        num_training_samples = _get_num_records(train_filenames)
    else:
        num_training_samples = global_batch_size
    train_idx_filenames = None
    if data_idx_dir is not None:
        filename_pattern = os.path.join(data_idx_dir, '%s-*')
        train_idx_filenames = sorted(tf.gfile.Glob(filename_pattern % 'train'))

    if iter_unit.lower() == 'epoch':
        nstep = num_training_samples * num_iter // global_batch_size
        decay_steps = nstep
    else:
        nstep = num_iter
        num_epochs = max(nstep * global_batch_size // num_training_samples, 1)
        decay_steps = 90 * num_training_samples // global_batch_size

    nstep_per_epoch = num_training_samples // global_batch_size

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(gpu_options=gpu_options)
    #config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.force_gpu_compatible = True # Force pinned memory
    config.intra_op_parallelism_threads = 1 # Avoid pool of Eigen threads
    config.inter_op_parallelism_threads = max(2, 40//hvd.size()-2)

    classifier = tf.estimator.Estimator(
        model_fn=_cnn_model_function,
        model_dir=log_dir,
        params={
            'model':         infer_func,
            'format':        image_format,
            'dtype' : tf.float16 if precision == 'fp16' else tf.float32,
            'momentum' : momentum,
            'learning_rate_init' : learning_rate_init,
            'learning_rate_power' : learning_rate_power,
            'decay_steps' : decay_steps,
            'weight_decay' : weight_decay,
            'loss_scale' : loss_scale,
            'larc_eta' : larc_eta,
            'larc_mode' : larc_mode,
            'deterministic' : deterministic,
            'n_classes':     1000,
            'use_dali': use_dali,
        },
        config=tf.estimator.RunConfig(
            tf_random_seed=2 * (1 + hvd.rank()) if deterministic else None,
            session_config=config,
            save_checkpoints_secs=checkpoint_secs if hvd.rank() == 0 else None,
            save_checkpoints_steps=nstep if hvd.rank() == 0 else None,
            keep_checkpoint_every_n_hours=3))

    print("Training")
    if not deterministic and not use_dali:
        num_preproc_threads = 10
    elif not deterministic and use_dali:
        num_preproc_threads = 2
    elif deterministic:
        num_preproc_threads = 1

    training_hooks = [hvd.BroadcastGlobalVariablesHook(0),
                      _PrefillStagingAreasHook()]
    if hvd.rank() == 0:
        training_hooks.append(
            _LogSessionRunHook(global_batch_size,
                               num_training_samples,
                               display_every))

    if data_dir is not None:
        input_func = lambda: nvutils.image_set(
            train_filenames, batch_size, image_height, image_width,
            training=True, distort_color=distort_color,
            deterministic=deterministic, num_threads=num_preproc_threads,
            use_dali=use_dali, idx_filenames=train_idx_filenames)
    else:
        input_func = lambda: nvutils.fake_image_set(
            batch_size, image_height, image_width)

    try:
        classifier.train(
            input_fn=input_func,
            max_steps=nstep,
            hooks=training_hooks)
    except KeyboardInterrupt:
        print("Keyboard interrupt")

def validate(infer_func, params):
    image_width = params['image_width']
    image_height = params['image_height']
    image_format = params['image_format']
    batch_size = params['batch_size']
    data_dir = params['data_dir']
    log_dir = params['log_dir']
    precision = params['precision']
    momentum = params['momentum']
    learning_rate_init = params['learning_rate_init']
    learning_rate_power = params['learning_rate_power']
    weight_decay = params['weight_decay']
    loss_scale = params['loss_scale']
    larc_eta = params['larc_eta']
    larc_mode = params['larc_mode']
    num_iter = params['num_iter']
    checkpoint_secs = params['checkpoint_secs']
    display_every = params['display_every']
    iter_unit = params['iter_unit']
    use_dali = params['use_dali']

    # Determinism is not fully supported by all TF ops.
    # Disabling until remaining wrinkles can be ironed out.
    deterministic = False
    if deterministic:
        tf.set_random_seed(2 * (1 + hvd.rank()))
        random.seed(3 * (1 + hvd.rank()))
        np.random.seed(2)

    log_dir  = None if log_dir  == "" else log_dir
    data_dir = None if data_dir == "" else data_dir
    if data_dir is None:
        raise ValueError("data_dir must be specified")
    if log_dir is None:
        raise ValueError("log_dir must be specified")

    filename_pattern = os.path.join(data_dir, '%s-*')
    eval_filenames  = sorted(tf.gfile.Glob(filename_pattern % 'validation'))

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.force_gpu_compatible = True # Force pinned memory
    config.intra_op_parallelism_threads = 1 # Avoid pool of Eigen threads
    config.inter_op_parallelism_threads = 40 // hvd.size() - 2 # HACK TESTING

    classifier = tf.estimator.Estimator(
        model_fn=_cnn_model_function,
        model_dir=log_dir,
        params={
            'model':         infer_func,
            'format':        image_format,
            'dtype' : tf.float16 if precision == 'fp16' else tf.float32,
            'momentum' : momentum,
            'learning_rate_init' : learning_rate_init,
            'learning_rate_power' : learning_rate_power,
            'decay_steps' : None,
            'weight_decay' : weight_decay,
            'loss_scale' : loss_scale,
            'larc_eta' : larc_eta,
            'larc_mode' : larc_mode,
            'deterministic' : deterministic,
            'n_classes':     1000,
            'use_dali': False,
        },
        config=tf.estimator.RunConfig(
            tf_random_seed=2 * (1 + hvd.rank()) if deterministic else None,
            session_config=config,
            save_checkpoints_secs=None,
            save_checkpoints_steps=None,
            keep_checkpoint_every_n_hours=3))

    if not deterministic and not use_dali:
        num_preproc_threads = 10
    elif not deterministic and use_dali:
        num_preproc_threads = 2
    elif deterministic:
        num_preproc_threads = 1

    if hvd.rank() == 0:
        print("Evaluating")
        try:
            eval_result = classifier.evaluate(
                input_fn=lambda: nvutils.image_set(
                    eval_filenames, batch_size, image_height, image_width,
                    training=False, distort_color=False,
                    deterministic=deterministic,
                    num_threads=num_preproc_threads))
            print('Top-1 accuracy:', eval_result['top1_accuracy']*100, '%')
            print('Top-5 accuracy:', eval_result['top5_accuracy']*100, '%')
        except KeyboardInterrupt:
            print("Keyboard interrupt")
