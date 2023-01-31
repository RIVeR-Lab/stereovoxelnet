from __future__ import print_function, division
import sys
sys.path.append('../')
import os
import torch.nn as nn
from utils import *
from utils.KittiColormap import *
from datasets.data_io import get_transform, read_all_lines
from PIL import Image
import pyvista
from moviepy.editor import VideoFileClip, clips_array


data_index = 24
list_filename = "../filenames/DS_test.txt"
lines = read_all_lines(list_filename)
splits = [line.split() for line in lines]
left_filenames = [x[0] for x in splits]
right_filenames = [x[1] for x in splits]
disp_filenames = [x[2] for x in splits]

def load_image(filename):
    return Image.open(filename).convert('RGB')

def load_disp(filename):
    data = Image.open(filename)
    data = np.array(data, dtype=np.float32) / 256.
    return data

datapath = "/home/chris/pl_ws/src/stereo_pl_nav/datasets/DS"
left_img = load_image(os.path.join(datapath, left_filenames[data_index]))
right_img = load_image(os.path.join(datapath, right_filenames[data_index]))
disparity = load_disp(os.path.join(datapath, disp_filenames[data_index]))
left_frame = np.asarray(left_img)
left_depth_rgb = left_frame[:, :, :3]

w, h = left_img.size
crop_w, crop_h = 880, 400

left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
disparity = disparity[h - crop_h:h, w - crop_w: w]
depth_rgb = np.transpose(np.asarray(left_img)[:, :, :3], (2, 0, 1))

processed = get_transform()
left_img = processed(left_img)
right_img = processed(right_img)

# Camera intrinsics and extrinsics
c_u = 4.556890e+2
c_v = 1.976634e+2
f_u = 1.003556e+3
f_v = 1.003556e+3
b_x = 0.0
b_y = 0.0
baseline = 0.54

def project_image_to_rect(uv_depth):
    ''' Input: nx3 first two channels are uv, 3rd channel
               is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    '''
    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
    y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
    pts_3d_rect = np.zeros((n, 3))
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    return pts_3d_rect

def project_image_to_velo(uv_depth):
    pts_3d_rect = project_image_to_rect(uv_depth)
    return pts_3d_rect


mask = disparity > 0
depth_gt = f_u * baseline / (disparity + 1. - mask)



mask = disparity > 0
rows, cols = depth_gt.shape
c, r = np.meshgrid(np.arange(cols), np.arange(rows))
points = np.stack([c, r, depth_gt])
points = points.reshape((3, -1))
points = points.T
points = points[mask.reshape(-1)]
cloud_gt = project_image_to_velo(points)


points_rgb = depth_rgb.reshape((3, -1)).T
points_rgb = points_rgb.astype(float)
points_rgb /= 255.

def create_voxel_grid(cloud_np, red_color=1.0):
    dist = np.linalg.norm(cloud_np, axis=1)
    dist_norm = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))
    dist_norm = 1 - dist_norm

    color = np.random.uniform(0, 1, size=(len(cloud_np), 3))
    color[:,1] = 0.0
    color[:,2] = 0.0
    # color[:,0] = dist_norm
    color[:,0] = red_color

    return cloud_np, color

def filter_cloud(cloud):
    min_mask = cloud >= [-16, -31, 0.0]
    max_mask = cloud <= [16, 1, 32]
    min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
    max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
    filter_mask = min_mask & max_mask
    filtered_cloud = cloud[filter_mask]
    return filtered_cloud

def calc_voxel_grid(filtered_cloud, grid_size):
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

sample_left = torch.Tensor(left_img)
sample_right = torch.Tensor(right_img)

sample_left = torch.unsqueeze(sample_left, dim=0)
sample_right = torch.unsqueeze(sample_right, dim=0)

from models.Voxel2D_hie import Voxel2D
voxel_model = Voxel2D(192, "voxel")
voxel_model = nn.DataParallel(voxel_model)
voxel_model.cuda()

ckpt_path = "../voxel.ckpt"
voxel_model.load_state_dict(torch.load(ckpt_path)['model'])
# calculate voxel cost volume disparity set
vox_cost_vol_disp_set = set()
max_disp = 192
# depth starting from voxel_size since 0 will cause issue
for z in np.arange(0.5, 32, 2.0):
    # get respective disparity
    d = f_u * baseline / z

    if d > max_disp:
        continue

    # real disparity -> disparity in feature map
    vox_cost_vol_disp_set.add(round(d/4))

vox_cost_vol_disps = list(vox_cost_vol_disp_set)
vox_cost_vol_disps = sorted(vox_cost_vol_disps)

tmp = []
for i in vox_cost_vol_disps:
    tmp.append(torch.unsqueeze(torch.Tensor([i]), 0))
vox_cost_vol_disps = tmp

def get_model_pred(model, sample_left, sample_right, voxel_disp):
    with torch.no_grad():
        return model(sample_left.cuda(), sample_right.cuda(), voxel_disp)[0]

voxel_pred = get_model_pred(voxel_model, sample_left, sample_right, vox_cost_vol_disps)

grid_sizes = [8, 16, 32, 64]

def get_level_points(level, red_color=1.0):
    vox_pred = voxel_pred[level].detach().cpu().numpy()[0]
    vox_pred[vox_pred < 0.5] = 0
    vox_pred[vox_pred >= 0.5] = 1
    voxel_size = 32/grid_sizes[level]
    offsets = np.array([int(16/voxel_size), int(31/voxel_size), 0])
    xyz_pred = np.asarray(np.where(vox_pred == 1)) # get back indexes of populated voxels
    cloud_pred = np.asarray([(pt-offsets)*voxel_size for pt in xyz_pred.T])
    points, colors = create_voxel_grid(cloud_pred, red_color)
    return points, colors

pl = pyvista.Plotter(off_screen=True)
pl.open_movie("hie.mp4")
# pl.open_gif("hie.gif")
pl.set_background("white")
pl.add_points(cloud_gt, scalars=points_rgb[mask.reshape(352000)], point_size=5, rgb=True)
pl.show(cpos=[(-0.3484873884891845, -3.5536092557145635, -5.746098824833765),
    (0.148353260725683, 5.5250374342419875, 60.67324901052562),
    (-0.007880808650864862, -0.9907485765920729, 0.13548118258927996)],return_cpos=True, screenshot=f"front.png", auto_close=False)

for _ in range(10):
    pl.write_frame()

colors = ["FD8D3C", "FC4E2A", "E31A1C", "B10026"]


for i in range(4):
    pl.clear()
    pl.add_points(cloud_gt, scalars=points_rgb[mask.reshape(352000)], point_size=5, rgb=True)
    points, _ = get_level_points(i, 0.25)
    for idx, p in enumerate(points):
        pl.add_mesh(pyvista.Cube(p, 32/grid_sizes[i], 32/grid_sizes[i], 32/grid_sizes[i]), show_edges=True, line_width=3, opacity=0.2, color=colors[i])

    print(pl.show(cpos=[(-0.3484873884891845, -3.5536092557145635, -5.746098824833765),
    (0.148353260725683, 5.5250374342419875, 60.67324901052562),
    (-0.007880808650864862, -0.9907485765920729, 0.13548118258927996)],return_cpos=True, screenshot=f"front-{i}.png", auto_close=False))
    for _ in range(10):
        pl.write_frame()

# print("going reverse")
# for i in range(4):
#     i = 3-i
#     pl.clear()
#     pl.add_points(cloud_gt, scalars=points_rgb[mask.reshape(352000)], point_size=5, rgb=True)
#     points, _ = get_level_points(i, 0.25)
#     for idx, p in enumerate(points):
#         pl.add_mesh(pyvista.Cube(p, 32/grid_sizes[i], 32/grid_sizes[i], 32/grid_sizes[i]), show_edges=True, line_width=3, opacity=0.2, color=colors[i])
#     print(pl.show(cpos=[(-0.3484873884891845, -3.5536092557145635, -5.746098824833765),
#     (0.148353260725683, 5.5250374342419875, 60.67324901052562),
#     (-0.007880808650864862, -0.9907485765920729, 0.13548118258927996)], auto_close=False))
#     for _ in range(10):
#         pl.write_frame()

pl.close()