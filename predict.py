import sys
import os
import chainer
from chainer.links.caffe import CaffeFunction
import numpy as np
from PIL import Image
from math import ceil
from tqdm import tqdm


class InceptionV3(CaffeFunction):
    def prepare(self, image):
        """Converts the given image to the np array for Inception-v3
        Note that you have to call this method before ``forward``
        because the pre-trained vgg model requires to resize the given
        image, covert the RGB to the BGR, and normalize to -1~1 range.

        Args:
            image (PIL.Image or np.ndarray): Input image.
                If an input is ``np.ndarray``, its shape must be
                ``(height, width)``, ``(height, width, channels)``,
                or ``(channels, height, width)``, and
                the order of the channels must be RGB.
            size (pair of ints): Size of converted images.
                If ``None``, the given image is not resized.
        Returns:
            np.ndarray: The converted output array.
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                if image.shape[0] == 1:
                    image = image[0, :, :]
                elif image.shape[0] == 3:
                    image = image.transpose((1, 2, 0))
            image = Image.fromarray(image.astype(np.uint8))
        image = image.convert('RGB')
        ratio = float(299) / min(image.size)
        image = image.resize(
            (ceil(ratio * image.size[0]), ceil(ratio * image.size[1])))
        image = np.asarray(image, dtype=chainer.get_dtype())

        image = center_crop(image, 299)
        image = preprocess_input(image)
        image = image[:, :, ::-1]
        image = image.transpose((2, 0, 1))
        return image


def center_crop(img, crop_size):
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    return img[yy: yy + crop_size, xx: xx + crop_size, :]


def preprocess_input(x):
    # loaded caffe is converted from official Keras implementation of
    # Inception-v3. So apply the same preprocessing as the official
    # impl. https://github.com/keras-team/keras-applications/blob/1.0.6/keras_applications/imagenet_utils.py#L45-L48
    x /= 127.5
    x -= 1.
    return x


# Load the model
model = InceptionV3(sys.argv[1])

if chainer.backends.intel64.is_ideep_available():
    setattr(chainer.config, 'use_ideep', 'auto')
    model.to_intel64()

BATCHSIZE = 32

results = []
for i in tqdm(list(range(1, 50000, BATCHSIZE))):
    imgs = []
    for j in range(i, min(i + BATCHSIZE, 50000)):
        img_path = os.path.join(
            sys.argv[2], "ILSVRC2012_val_%08d.JPEG" % j)
        img = Image.open(img_path)
        img = model.prepare(img)
        imgs.append(img)
    imgs = np.asarray(imgs)

    assert imgs.shape[1] == 3
    assert imgs.shape[2] == 299
    assert imgs.shape[3] == 299

    with chainer.using_config('enable_backprop', False), chainer.using_config('train', False), chainer.using_config('type_check', False):
        imgs = chainer.Variable(imgs)
        y, = model(inputs={'data': imgs}, outputs=['prob'])
        results.append(chainer.cuda.to_cpu(y.data))

results = np.concatenate(results, axis=0)
# get top 5 indices
top5 = np.argsort(results)[:, -5:][:, ::-1]
# save as ImageNet prediction format
np.savetxt(sys.argv[3], top5, delimiter=" ", fmt='%d')
