import os
import numpy as np
import skimage.measure
from PIL import Image
import torch
from torch.utils.data import Dataset
from .data_io import get_transform, read_all_lines, pfm_imread


class VoxelDataset(Dataset):

    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(
            list_filename)
        self.training = training

        # Camera intrinsics
        # 15mm images have different focals
        self.c_u = 479.5
        self.c_v = 269.5
        self.f_u = 1050.0
        self.f_v = 1050.0
        self.baseline = 0.1
        self.voxel_size = 0.05

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = []
        right_images = []
        disp_images = []
        for x in splits:
            if "15mm_focallength" in x[0]:
                continue
            if "funnyworld" in x[0]:
                continue
            left_images.append(x[0])
            right_images.append(x[1])
            disp_images.append(x[2])
        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / \
            self.f_u + self.baseline
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return pts_3d_rect

    def calc_cloud(self, disp_est, depth):
        mask = disp_est > 0
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, depth])
        points = points.reshape((3, -1))
        points = points.T
        points = points[mask.reshape(-1)]
        cloud = self.project_image_to_velo(points)
        return cloud

    def filter_cloud(self, cloud):
        min_mask = cloud >= [-1.6, -3.0, 0.0]
        max_mask = cloud <= [1.6, 0.2, 3.2]
        min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
        max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
        filter_mask = min_mask & max_mask
        filtered_cloud = cloud[filter_mask]
        return filtered_cloud

    def calc_voxel_grid(self, filtered_cloud, voxel_size):
        # quantized point values, here you will loose precision
        xyz_q = np.floor(np.array(filtered_cloud/voxel_size)).astype(int)
        vox_grid = np.zeros((int(3.2/voxel_size), int(3.2/voxel_size),
                            int(3.2/voxel_size)))  # Empty voxel grid
        offsets = np.array([32, 60, 0])
        xyz_offset_q = xyz_q+offsets
        # Setting all voxels containitn a points equal to 1
        vox_grid[xyz_offset_q[:, 0],
                 xyz_offset_q[:, 1], xyz_offset_q[:, 2]] = 1

        # get back indexes of populated voxels
        xyz_v = np.asarray(np.where(vox_grid == 1))
        cloud_np = np.asarray([(pt-offsets)*voxel_size for pt in xyz_v.T])
        return vox_grid, cloud_np

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(
            self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(
            self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(
            self.datapath, self.disp_filenames[index]))

        if "15mm_focallength" in self.left_filenames[index]:
            self.f_u = 450.0
            self.f_v = 450.0
        else:
            self.f_u = 1050.0
            self.f_v = 1050.0

        if self.training:
            w, h = left_img.size
            # crop_w, crop_h = 512, 256

            # x1 = random.randint(0, w - crop_w)
            # y1 = random.randint(0, h - crop_h)

            # # random crop
            # left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            # right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            # disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            crop_w, crop_h = 960, 512

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            # calcualte depth for ground truth disparity map
            mask = disparity > 0
            depth_gt = self.f_u * self.baseline / (disparity + 1. - mask)
            vox_grid_gt = torch.zeros(
                (int(2.4/self.voxel_size)+1, int(3.2/self.voxel_size), int(6.4/self.voxel_size)))
            try:
                cloud_gt = self.calc_cloud(disparity, depth_gt)
                filtered_cloud_gt = self.filter_cloud(cloud_gt)
                vox_grid_gt, cloud_np_gt = self.calc_voxel_grid(
                    filtered_cloud_gt, voxel_size=self.voxel_size)
                vox_grid_gt = torch.from_numpy(vox_grid_gt)
            except Exception as e:
                pass

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "voxel_grid": vox_grid_gt,
                    "left_filename": self.left_filenames[index]}
        else:
            w, h = left_img.size
            crop_w, crop_h = 960, 512

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            # calcualte depth for ground truth disparity map
            mask = disparity > 0
            depth_gt = self.f_u * self.baseline / (disparity + 1. - mask)
            vox_grid_gt = torch.zeros(
                (int(2.4/self.voxel_size)+1, int(3.2/self.voxel_size), int(6.4/self.voxel_size)))
            try:
                cloud_gt = self.calc_cloud(disparity, depth_gt)
                filtered_cloud_gt = self.filter_cloud(cloud_gt)
                vox_grid_gt, cloud_np_gt = self.calc_voxel_grid(
                    filtered_cloud_gt, voxel_size=self.voxel_size)
                vox_grid_gt = torch.from_numpy(vox_grid_gt)
            except Exception as e:
                pass

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "voxel_grid": vox_grid_gt,
                    "top_pad": 0,
                    "right_pad": 0,
                    "left_filename": self.left_filenames[index]}


class VoxelKITTIDataset(Dataset):

    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(
            list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

        # Camera intrinsics
        # 15mm images have different focals
        self.c_u = 6.071928e+02
        self.c_v = 1.852157e+02
        self.f_u = 7.188560e+02
        self.f_v = 7.188560e+02
        self.baseline = 0.54
        self.voxel_size = 0.5

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / \
            self.f_u + self.baseline
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return pts_3d_rect

    def calc_cloud(self, disp_est, depth):
        mask = disp_est > 0
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, depth])
        points = points.reshape((3, -1))
        points = points.T
        points = points[mask.reshape(-1)]
        cloud = self.project_image_to_velo(points)
        return cloud

    def filter_cloud(self, cloud):
        min_mask = cloud >= [-16, -28, 0.0]
        max_mask = cloud <= [16, 4.0, 32]
        min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
        max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
        filter_mask = min_mask & max_mask
        filtered_cloud = cloud[filter_mask]
        return filtered_cloud

    def calc_voxel_grid(self, filtered_cloud, voxel_size):
        # quantized point values, here you will loose precision
        xyz_q = np.floor(np.array(filtered_cloud/voxel_size)).astype(int)
        # Empty voxel grid
        vox_grid = np.zeros(
            (int(32/voxel_size), int(32/voxel_size), int(32/voxel_size)))
        offsets = np.array([32, 56, 0])
        xyz_offset_q = xyz_q+offsets
        # Setting all voxels containitn a points equal to 1
        vox_grid[xyz_offset_q[:, 0],
                 xyz_offset_q[:, 1], xyz_offset_q[:, 2]] = 1

        # get back indexes of populated voxels
        xyz_v = np.asarray(np.where(vox_grid == 1))
        cloud_np = np.asarray([(pt-offsets)*voxel_size for pt in xyz_v.T])
        return vox_grid, cloud_np

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(
            self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(
            self.datapath, self.right_filenames[index]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(
                self.datapath, self.disp_filenames[index]))
        else:
            disparity = None

        w, h = left_img.size

        # normalize
        processed = get_transform()
        left_img = processed(left_img).numpy()
        right_img = processed(right_img).numpy()

        # pad to size 1248x384
        top_pad = 384 - h
        right_pad = 1248 - w
        assert top_pad > 0 and right_pad > 0
        # pad images
        left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0),
                              (0, right_pad)), mode='constant', constant_values=0)
        right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                               constant_values=0)
        # pad disparity gt
        if disparity is not None:
            assert len(disparity.shape) == 2
            disparity = np.lib.pad(
                disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

        # calcualte depth for ground truth disparity map
        mask = disparity > 0
        depth_gt = self.f_u * self.baseline / (disparity + 1. - mask)
        vox_grid_gt = np.zeros(
            (int(32/self.voxel_size), int(32/self.voxel_size), int(32/self.voxel_size)))
        try:
            cloud_gt = self.calc_cloud(disparity, depth_gt)
            filtered_cloud_gt = self.filter_cloud(cloud_gt)
            vox_grid_gt, cloud_np_gt = self.calc_voxel_grid(
                filtered_cloud_gt, voxel_size=self.voxel_size)
            vox_grid_gt = torch.from_numpy(vox_grid_gt)
        except Exception as e:
            pass

        return {"left": left_img,
                "right": right_img,
                "disparity": disparity,
                "voxel_grid": vox_grid_gt,
                "top_pad": 0,
                "right_pad": 0,
                "left_filename": self.left_filenames[index]}


class VoxelDSDataset(Dataset):

    def __init__(self, datapath, list_filename, training, transform=True, lite=False):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(
            list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

        # Camera intrinsics
        # 15mm images have different focals
        self.c_u = 4.556890e+2
        self.c_v = 1.976634e+2
        self.f_u = 1.003556e+3
        self.f_v = 1.003556e+3
        self.baseline = 0.54
        self.voxel_size = 0.5
        self.grid_sizes = [8, 16, 32, 64]
        # set the maximum perception depth
        self.max_depth = 32
        self.transform = transform

        # calculate voxel cost volume disparity set
        vox_cost_vol_disp_set = set()
        max_disp = 192
        # depth starting from voxel_size since 0 will cause issue
        if not lite:
            for z in np.arange(self.voxel_size, self.max_depth, self.voxel_size*4):
                # get respective disparity
                d = self.f_u * self.baseline / z
                if d > max_disp:
                    continue
                # real disparity -> disparity in feature map
                vox_cost_vol_disp_set.add(round(d/4))
        else:
            for z in np.arange(self.voxel_size, self.max_depth, self.voxel_size):
                # get respective disparity
                d = self.f_u * self.baseline / z
                if d > max_disp:
                    continue
                # real disparity -> disparity in feature map
                vox_cost_vol_disp_set.add(round(d/8))

        self.vox_cost_vol_disps = list(vox_cost_vol_disp_set)
        self.vox_cost_vol_disps = sorted(self.vox_cost_vol_disps)
        # if not lite:
        #     self.vox_cost_vol_disps = self.vox_cost_vol_disps[1:]

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / \
            self.f_u + self.baseline
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return pts_3d_rect

    def calc_cloud(self, disp_est, depth):
        mask = disp_est > 0
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, depth])
        points = points.reshape((3, -1))
        points = points.T
        points = points[mask.reshape(-1)]
        cloud = self.project_image_to_velo(points)
        return cloud

    def filter_cloud(self, cloud):
        min_mask = cloud >= [-16, -31, 0.0]
        max_mask = cloud <= [16, 1, self.max_depth]
        min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
        max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
        filter_mask = min_mask & max_mask
        filtered_cloud = cloud[filter_mask]
        return filtered_cloud

    def calc_voxel_grid(self, filtered_cloud, grid_size):
        voxel_size = 32/grid_size
        # quantized point values, here you will loose precision
        xyz_q = np.floor(np.array(filtered_cloud/voxel_size)).astype(int)
        # Empty voxel grid
        vox_grid = np.zeros((grid_size, grid_size, grid_size))
        offsets = np.array([int(16/voxel_size), int(31/voxel_size), 0])
        xyz_offset_q = xyz_q+offsets
        # Setting all voxels containitn a points equal to 1
        vox_grid[xyz_offset_q[:, 0],
                 xyz_offset_q[:, 1], xyz_offset_q[:, 2]] = 1

        # get back indexes of populated voxels
        xyz_v = np.asarray(np.where(vox_grid == 1))
        cloud_np = np.asarray([(pt-offsets)*voxel_size for pt in xyz_v.T])
        return vox_grid, cloud_np

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(
            self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(
            self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(
            self.datapath, self.disp_filenames[index]))

        w, h = left_img.size
        crop_w, crop_h = 880, 400

        processed = get_transform()

        if self.transform:
            if w < crop_w:
                left_img = processed(left_img).numpy()
                right_img = processed(right_img).numpy()

                left_img = np.lib.pad(
                    left_img, ((0, 0), (0, 0), (0, crop_w-w)), mode='constant', constant_values=0)
                right_img = np.lib.pad(
                    right_img, ((0, 0), (0, 0), (0, crop_w-w)), mode='constant', constant_values=0)
                disparity = np.lib.pad(
                    disparity, ((0, 0), (0, crop_w-w)), mode='constant', constant_values=0)

                left_img = torch.Tensor(left_img)
                right_img = torch.Tensor(right_img)
            else:
                left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
                right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
                disparity = disparity[h - crop_h:h, w - crop_w: w]

                left_img = processed(left_img)
                right_img = processed(right_img)
        else:
            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            left_img = np.asarray(left_img)
            right_img = np.asarray(right_img)

        # calcualte depth for ground truth disparity map
        mask = disparity > 0
        depth_gt = self.f_u * self.baseline / (disparity + 1. - mask)

        cloud_gt = self.calc_cloud(disparity, depth_gt)
        filtered_cloud_gt = self.filter_cloud(cloud_gt)
        all_vox_grid_gt = []
        try:
            for grid_size in self.grid_sizes:
                vox_grid_gt = np.zeros((grid_size, grid_size, grid_size))

                vox_grid_gt, cloud_np_gt = self.calc_voxel_grid(
                    filtered_cloud_gt, grid_size=grid_size)
                vox_grid_gt = torch.from_numpy(vox_grid_gt)

                all_vox_grid_gt.append(vox_grid_gt)
        except Exception as e:
            pass

        return {"left": left_img,
                "right": right_img,
                "disparity": disparity,
                "voxel_grid": all_vox_grid_gt,
                "vox_cost_vol_disps": self.vox_cost_vol_disps,
                "top_pad": 0,
                "right_pad": 0,
                "left_filename": self.left_filenames[index]}


class VoxelISECDataset(Dataset):

    def __init__(self, datapath, list_filename, training, transform=True, lite=False):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(
            list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

        # Camera intrinsics
        # 15mm images have different focals
        self.c_u = 447.59914779663086
        self.c_v = 255.83612823486328
        self.f_u = 365.68
        self.f_v = 365.68
        self.baseline = 0.12
        self.voxel_size = 0.1
        # set the maximum perception depth
        self.max_depth = 6.4
        self.transform = transform
        self.grid_sizes = [8, 16, 32, 64]

        # calculate voxel cost volume disparity set
        vox_cost_vol_disp_set = set()
        max_disp = 192
        # depth starting from voxel_size since 0 will cause issue
        if not lite:
            for z in np.arange(self.voxel_size, self.max_depth, self.voxel_size*2):
                # get respective disparity
                d = self.f_u * self.baseline / z

                if d > max_disp:
                    continue

                # real disparity -> disparity in feature map
                vox_cost_vol_disp_set.add(round(d/4))
        else:
            for z in np.arange(self.voxel_size, self.max_depth, self.voxel_size/2):
                # get respective disparity
                d = self.f_u * self.baseline / z

                if d > max_disp:
                    continue

                # real disparity -> disparity in feature map
                vox_cost_vol_disp_set.add(round(d/8))

        self.vox_cost_vol_disps = list(vox_cost_vol_disp_set)
        self.vox_cost_vol_disps = sorted(self.vox_cost_vol_disps)
        if not lite:
            self.vox_cost_vol_disps = self.vox_cost_vol_disps[1:]

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = np.load(os.path.join(
            self.datapath, self.left_filenames[index]))
        right_img = np.load(os.path.join(
            self.datapath, self.right_filenames[index]))
        vox_grid_gt = np.load(os.path.join(
            self.datapath, self.disp_filenames[index]))

        h, w, _ = left_img.shape
        crop_w, crop_h = 880, 400

        processed = get_transform()

        if self.transform:
            if w < crop_w:
                left_img = processed(left_img).numpy()
                right_img = processed(right_img).numpy()

                left_img = np.lib.pad(
                    left_img, ((0, 0), (0, 0), (0, crop_w-w)), mode='constant', constant_values=0)
                right_img = np.lib.pad(
                    right_img, ((0, 0), (0, 0), (0, crop_w-w)), mode='constant', constant_values=0)

                left_img = torch.Tensor(left_img)
                right_img = torch.Tensor(right_img)
            else:
                left_img = processed(left_img)
                right_img = processed(right_img)

        # downsample 64x64x64 to 32, 16, 8
        all_vox_grid_gt = [vox_grid_gt]
        while len(all_vox_grid_gt) != len(self.grid_sizes):
            all_vox_grid_gt.append(skimage.measure.block_reduce(all_vox_grid_gt[-1], 2, np.max))
        all_vox_grid_gt.reverse()

        return {"left": left_img,
                "right": right_img,
                "voxel_grid": all_vox_grid_gt,
                "vox_cost_vol_disps": self.vox_cost_vol_disps,
                "top_pad": 0,
                "right_pad": 0,
                "left_filename": self.left_filenames[index]}
