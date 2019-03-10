import os
import visdom
import numpy as np
from PIL import Image
import torch


class Visualizer(object):
    def __init__(self, server, port, env='default'):
        """
        :param server: your server IP
        :param port: your visdom port
        :param env: set an Environment for you experimentation
        """
        self.vis = visdom.Visdom(
            server=server,
            port=port,
            env=env,
            raise_exceptions=True)
        self.index = {}

    def display_current_results(self, visuals, epoch, img=True):
        idx = 1
        for label, image in visuals.items():
            if img:
                image_numpy = tensor2im(image)
            else:
                image_numpy = decode_labels(image)[0]
            self.vis.image(image_numpy.transpose([2, 0, 1]),
                           opts=dict(title=label+"_epoch"+str(epoch),
                                     caption=None),
                           win=1 + idx)
            idx += 1

    def plot_many_stack(self, data_dic, split=False,
                        xlabel='epoch', ylabel='loss'):
        """
        use like this:
        vis.plot_many_stack({'train_loss': loss_meter.value()[0],
                             'test_loss': loss_meter1.value()[0]},
                            split=False)
        :param data_dic:
        :param split:
        :return:
        """
        name = list(data_dic.keys())
        name_total = " ".join(name)

        x = self.index.get(name_total, 0)
        val = list(data_dic.values())

        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))

        if len(val) is not 1 and split:
            for i in range(len(val)):
                self.vis.line(
                    Y=y[:, i], X=np.ones(y[:, i].shape) * x,
                    win=str(name_total[i]),
                    opts=dict(legend=[name[i]],
                              title=str(name[i]),
                              xlabel=xlabel,
                              ylabel=ylabel),
                    update=None if x == 0 else 'append'
                )
        else:

            self.vis.line(
                Y=y, X=np.ones(y.shape) * x,
                win=str(name_total),
                opts=dict(legend=name,
                          title=name_total,
                          xlabel=xlabel,
                          ylabel=ylabel),
                update=None if x == 0 else 'append'
            )
        self.index[name_total] = x + 1


def print_current_losses(txt_file, epoch, iter, losses_str):
    message = '(epoch: %d, iters: %d) ' % (epoch, iter)
    message += losses_str

    print(message)
    with open(txt_file, "a") as log_file:
        log_file.write('%s\n' % message)


def mkdirs(paths):
    """
    :param paths: str or str-list
    :return: None
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


label_colours = [
    (0,0,0), (128,0,0), (255,0,0), (0,85,0),
    (170,0,51), (255,85,0), (0,0,85), (0,119,221),
    (85,85,0), (0,85,85), (85,51,0), (52,86,128),
    (0,128,0), (0,0,255), (51,170,221), (0,255,255),
    (85,255,170), (170,255,85), (255,255,0), (255,170,0)
]


def decode_labels(data, n_classes=20):
    """Decode batch of segmentation masks.
    用于将网络输出来的结果转换成可视的RGB图像像素范围.
    输出的结果可直接根据第一个维度遍历存储（使用CV2要注意转为BGR格式）
    data: [batch, C, 128, 128] -- torch or numpy
    out: [batch, 128, 128, 3] -- numpy
    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    (_, b) = data.max(dim=1)
    mask = b.unsqueeze(3)
    n, h, w, c = mask.shape
    num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i_ in range(num_images):
        img = Image.new('RGB', (len(mask[i_, 0]), len(mask[i_])))
        pixels = img.load()
        for j_, j in enumerate(mask[i_, :, :, 0]):
            for k_, k in enumerate(j):
                if k < n_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i_] = np.array(img)
    return outputs
