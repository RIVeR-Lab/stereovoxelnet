from __future__ import print_function, division
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import copy
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import coloredlogs, logging
from mpl_toolkits.mplot3d import Axes3D

log = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=log)

def make_iterative_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


@make_iterative_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type for tensor2float")


@make_iterative_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.cpu().numpy()
    else:
        raise NotImplementedError("invalid input type for tensor2numpy")


@make_iterative_func
def check_allfloat(vars):
    assert isinstance(vars, float)


def save_scalars(logger: SummaryWriter, mode_tag, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for tag, values in scalar_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            scalar_name = '{}/{}'.format(mode_tag, tag)
            # if len(values) > 1:
            scalar_name = scalar_name + "_" + str(idx)
            logger.add_scalar(scalar_name, value, global_step)


def save_images(logger: SummaryWriter, mode_tag, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)
    for tag, values in images_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            if len(value.shape) == 3:
                value = value[:, np.newaxis, :, :]
            value = value[:1]
            value = torch.from_numpy(value)

            image_name = '{}/{}'.format(mode_tag, tag)
            if len(values) > 1:
                image_name = image_name + "_" + str(idx)
            logger.add_image(image_name, vutils.make_grid(value, padding=0, nrow=1, normalize=True, scale_each=True),
                             global_step)

def voxel_to_pc(voxel, voxel_size = 0.05, offsets = np.array([24, 20, 0])):
    xyz = np.asarray(np.where(voxel == 1)) # get back indexes of populated voxels
    cloud = np.asarray([(pt-offsets)*voxel_size for pt in xyz.T])
    return cloud

def save_voxel(logger: SummaryWriter, mode_tag, vox_grid, global_step, logdir, gt_only):
    vox_pred = torch.detach(vox_grid[0]).cpu().numpy()
    vox_gt = torch.detach(vox_grid[1]).cpu().numpy()

    vox_pred[vox_pred < 0.5] = 0
    vox_pred[vox_pred >= 0.5] = 1

    voxel_size = 0.5
    offsets = np.array([32, 62, 0])

    cloud_pred = voxel_to_pc(vox_pred, voxel_size=voxel_size, offsets=offsets)
    cloud_gt = voxel_to_pc(vox_gt, voxel_size=voxel_size, offsets=offsets)
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(cloud_gt[:,0],cloud_gt[:,2],cloud_gt[:,1],marker="o")
        if not gt_only:
            ax.scatter(cloud_pred[:,0],cloud_pred[:,2],cloud_pred[:,1],marker="^")
        # plt.savefig("scatter.png")
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = np.transpose(data, (2,0,1))
        logger.add_image(mode_tag+"/plot", data, global_step)

    except Exception as e:
        log.error(f"Error occured when saving voxel: {e}. cloud_pred shape {cloud_pred.shape}. cloud_gt shape {cloud_gt.shape}")
    plt.close(fig)

    


def adjust_learning_rate(optimizer, epoch, base_lr, lrepochs):
    splits = lrepochs.split(':')
    assert len(splits) == 2

    # parse the epochs to downscale the learning rate (before :)
    downscale_epochs = [int(eid_str) for eid_str in splits[0].split(',')]
    # parse downscale rate (after :)
    downscale_rate = float(splits[1])
    print("Downscale learning rate at epochs: {}, downscale rate: {}".format(downscale_epochs, downscale_rate))

    lr = base_lr
    for eid in downscale_epochs:
        if epoch >= eid:
            lr /= downscale_rate
        else:
            break
    print("Setting learning rate to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class AverageMeter(object):
    def __init__(self):
        self.sum_value = 0.
        self.count = 0

    def update(self, x):
        check_allfloat(x)
        self.sum_value += x
        self.count += 1

    def mean(self):
        return self.sum_value / self.count


class AverageMeterDict(object):
    def __init__(self):
        self.data = None
        self.count = 0

    def update(self, x):
        check_allfloat(x)
        self.count += 1
        if self.data is None:
            self.data = copy.deepcopy(x)
        else:
            for k1, v1 in x.items():
                if isinstance(v1, float):
                    self.data[k1] += v1
                elif isinstance(v1, tuple) or isinstance(v1, list):
                    for idx, v2 in enumerate(v1):
                        self.data[k1][idx] += v2
                else:
                    assert NotImplementedError("error input type for update AvgMeterDict")

    def mean(self):
        @make_iterative_func
        def get_mean(v):
            return v / float(self.count)

        return get_mean(self.data)
