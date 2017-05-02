from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime

import numpy as np
import tensorflow as tf

layers = tf.contrib.layers


def build_graph(grid_dim=100):
    """
    Use NCHW format.
    
    :param grid_dim: 
    :return: 
    """
    with tf.name_scope("input_position") as scope:
        input_positions = tf.placeholder(tf.float32, [grid_dim, grid_dim], name=scope)
        pos = tf.reshape(input_positions, [1, grid_dim, grid_dim, 1])

    with tf.name_scope("input_velocity") as scope:
        input_velocities = tf.placeholder(tf.float32, [grid_dim, grid_dim], name=scope)

    with tf.name_scope("laplacian"):
        weights = tf.placeholder(tf.float32, [3, 3, 1, 1], name="laplacian_weights")
        laplacian = tf.nn.conv2d(pos, weights, strides=[1, 1, 1, 1], padding='SAME')
        laplacian = tf.reshape(laplacian, [grid_dim, grid_dim])

    with tf.name_scope("update_velocities"):
        tmp = input_velocities + laplacian

    with tf.name_scope("update_positions"):
        pos = tf.reshape(pos, [grid_dim, grid_dim])
        out = pos + tmp

    return {
        "input_positions": input_positions,
        "input_velocities": input_velocities,
        "output_positions": out,
        "output_velocities": tmp,
        "weights": weights,
    }


def mkdir(path):
    if not os.path.exists(path):
        # TODO: log warning that directory was created
        os.makedirs(path)
    return path


def main(grid_dim, logdir=None):

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    logdir = logdir or os.path.join("/logs", subdir)
    mkdir(logdir)

    # laplacian weights
    scalar = 0.2
    weights = np.zeros([1, 1, 3, 3])
    weights[0, 0, 0, :] = scalar * np.array([0., 1., 0.])
    weights[0, 0, 1, :] = scalar * np.array([1., -4., 1.])
    weights[0, 0, 2, :] = scalar * np.array([0., 1., 0.])
    weights = np.transpose(weights, [2, 3, 0, 1])

    initial_p = np.zeros([grid_dim, grid_dim])
    initial_v = np.zeros([grid_dim, grid_dim])
    initial_p[10, 10] = 42.

    with tf.Graph().as_default() as graph:
        global_step = tf.Variable(0, name="global_step", trainable=False)
        tensor_map = build_graph(grid_dim=grid_dim)

        with tf.name_scope("viz_velocities"):
            output_images = tf.reshape(tensor_map["output_velocities"], [1, grid_dim, grid_dim, 1])
            tf.summary.image("velocities", output_images)
            summary_op = tf.summary.merge_all()

        with tf.control_dependencies([summary_op]):
            with tf.name_scope("update_step"):
                step_op = tf.assign(global_step, global_step + 1)

        # write compute graph
        summary_writer = tf.summary.FileWriter(logdir, graph)
        summary_writer.flush()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            output_positions = initial_p
            output_velocities = initial_v
            for _ in xrange(20):
                _, summary, step, output_positions, output_velocities = sess.run(
                    [step_op, summary_op, global_step, tensor_map["output_positions"], tensor_map["output_velocities"]],
                    feed_dict={
                        tensor_map["input_positions"]: output_positions,
                        tensor_map["input_velocities"]: output_velocities,
                        tensor_map["weights"]: weights,
                    }
                )
                summary_writer.add_summary(summary, step)


if __name__ == "__main__":
    main(grid_dim=100)
