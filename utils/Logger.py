# -*-coding:utf-8-*-
from __future__ import absolute_import
import os
import sys
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class TFLogger(object):
    def __init__(self, log_dir=None):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir=log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag=tag,
                                scalar_value = value,
                                global_step = step)
        self.writer.flush()

    # def image_summary(self, tag, images, step):
    #     """Log a list of images."""
    #
    #     img_summaries = []
    #     for i, img in enumerate(images):
    #         # Write the image to a string
    #         s = BytesIO()
    #         scipy.misc.toimage(img).save(s, format="png")
    #
    #         # Create an Image object
    #         img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
    #                                    height=img.shape[0],
    #                                    width=img.shape[1])
    #         # Create a Summary value
    #         img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))
    #
    #     # Create and write Summary
    #     # summary = tf.Summary(value=img_summaries)
    #     # self.writer.add_summary(summary, step)
    #     with self.writer.as_default(step=step):
    #         tf.summary.histogram(name='%s/%d' % (tag, i), data=img_summaries)
    #     self.writer.flush()
    #
    # def histo_summary(self, tag, values, step, bins=1000):
    #     """Log a histogram of the tensor of values."""
    #
    #     # Create a histogram using numpy
    #     counts, bin_edges = np.histogram(values, bins=bins)
    #
    #     # Fill the fields of the histogram proto
    #     hist = tf.HistogramProto()
    #     hist.min = float(np.min(values))
    #     hist.max = float(np.max(values))
    #     hist.num = int(np.prod(values.shape))
    #     hist.sum = float(np.sum(values))
    #     hist.sum_squares = float(np.sum(values ** 2))
    #
    #     # Drop the start of the first bin
    #     bin_edges = bin_edges[1:]
    #
    #     # Add bin edges and counts
    #     for edge in bin_edges:
    #         hist.bucket_limit.append(edge)
    #     for c in counts:
    #         hist.bucket.append(c)
    #
    #     # Create and write Summary
    #     # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    #     # self.writer.add_summary(summary, step)
    #     with self.writer.as_default(step=step):
    #         tf.summary.histogram(name=tag, data=hist)
    #     self.writer.flush()

    def close(self):
        self.writer.close()
