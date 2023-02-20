from __future__ import print_function, division
import sys
sys.path.append('../')
import os
import torch.nn as nn
from utils import *
import cv2
from datasets.data_io import get_transform, read_all_lines
from PIL import Image
import pyvista
import math
from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip


mov_r = 2.0
mov_step = 0.2 #The lower this value the higher quality the circle is with more points generated

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

left_img = load_image("./demo/left.jpg")
right_img = load_image("./demo/right.jpg")
disparity = load_disp("./demo/disparity.png")
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

def sgbm():
    print("visualizing SGBM")
    #Note: disparity range is tuned according to specific parameters obtained through trial and error. 
    win_size = 5
    min_disp = -1
    max_disp = 191 #min_disp * 9
    num_disp = max_disp - min_disp # Needs to be divisible by 16
    #Create Block matching object. 
    sgbm = cv2.StereoSGBM_create(minDisparity= min_disp,
    numDisparities = num_disp,
    blockSize = 5,
    uniquenessRatio = 5,
    speckleWindowSize = 5,
    speckleRange = 5,
    disp12MaxDiff = 1,
    P1 = 8*3*win_size**2,#8*3*win_size**2,
    P2 =32*3*win_size**2) #32*3*win_size**2)
    left_img = load_image("./demo/left.jpg")
    right_img = load_image("./demo/right.jpg")
    sgbm_disparity = sgbm.compute(cv2.cvtColor(np.asarray(left_img), cv2.COLOR_BGR2GRAY),cv2.cvtColor(np.asarray(right_img), cv2.COLOR_BGR2GRAY))
    sgbm_disparity[sgbm_disparity < 0] = 0
    sgbm_disparity = sgbm_disparity/3040*192.
    disp_est = sgbm_disparity
    mask = disp_est > 0
    depth = f_u * baseline / (disp_est + 1. - mask)
    mask = disp_est > 0
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = project_image_to_velo(points)

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

    voxel_size = 0.5
    min_mask = cloud >= [-16,-31,0.0]
    max_mask = cloud <= [16,1,32]
    min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
    max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
    filter_mask = min_mask & max_mask
    filtered_cloud = cloud[filter_mask]

    xyz_q = np.floor(np.array(filtered_cloud/voxel_size)).astype(int) # quantized point values, here you will loose precision
    vox_grid = np.zeros((int(32/voxel_size), int(32/voxel_size), int(32/voxel_size))) #Empty voxel grid
    offsets = np.array([32, 62, 0])
    xyz_offset_q = xyz_q+offsets
    vox_grid[xyz_offset_q[:,0],xyz_offset_q[:,1],xyz_offset_q[:,2]] = 1 # Setting all voxels containitn a points equal to 1

    xyz_v = np.asarray(np.where(vox_grid == 1)) # get back indexes of populated voxels
    cloud_np = np.asarray([(pt-offsets)*voxel_size for pt in xyz_v.T])

    voxel_grid = create_voxel_grid(cloud_np)

    pl = pyvista.Plotter(off_screen=True)
    pl.set_background("white")
    pl.add_points(cloud_gt, scalars=points_rgb[mask.reshape(352000)], point_size=5, rgb=True)
    pl.show(cpos=[(-0.3484873884891845, -3.5536092557145635, -5.746098824833765),
        (0.148353260725683, 5.5250374342419875, 60.67324901052562),
        (-0.007880808650864862, -0.9907485765920729, 0.13548118258927996)],return_cpos=True, screenshot=f"output/{data_index}-bg.png", auto_close=False)

    for idx, p in enumerate(cloud_np):
        pl.add_mesh(pyvista.Cube(p, voxel_size, voxel_size, voxel_size), show_edges=True, line_width=3, opacity=0.2, color="B10026")
    
    pl.show(cpos=[(-0.3484873884891845, -3.5536092557145635, -5.746098824833765),
        (0.148353260725683, 5.5250374342419875, 60.67324901052562),
        (-0.007880808650864862, -0.9907485765920729, 0.13548118258927996)],return_cpos=True, screenshot=f"output/{data_index}-sgbm.png", auto_close=False)
    
    pl.open_movie(f"output-mov/{data_index}-sgbm.mov")
    r = mov_r
    #The lower this value the higher quality the circle is with more points generated
    stepSize = mov_step
    t = 0
    while t < 2 * math.pi:
        pl.camera.position = (-0.3484873884891845 + r * math.cos(t), -3.5536092557145635 + r * math.sin(t), -5.746098824833765)
        pl.camera.focal_point = (0.148353260725683, 5.5250374342419875, 60.67324901052562)
        pl.camera.up = (-0.007880808650864862, -0.9907485765920729, 0.13548118258927996)
        pl.write_frame()
        t += stepSize

    pl.close()

def voxel():
    print("Visualizing Voxel")
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
    pl.set_background("white")
    pl.add_points(cloud_gt, scalars=points_rgb[mask.reshape(352000)], point_size=5, rgb=True)


    colors = ["FD8D3C", "FC4E2A", "E31A1C", "B10026"]

    i=3
    points, _ = get_level_points(i, 0.25)
    for idx, p in enumerate(points):
        pl.add_mesh(pyvista.Cube(p, 32/grid_sizes[i], 32/grid_sizes[i], 32/grid_sizes[i]), show_edges=True, line_width=3, opacity=0.2, color=colors[i])

    pl.show(cpos=[(-0.3484873884891845, -3.5536092557145635, -5.746098824833765),
    (0.148353260725683, 5.5250374342419875, 60.67324901052562),
    (-0.007880808650864862, -0.9907485765920729, 0.13548118258927996)],return_cpos=True, screenshot=f"output/{data_index}-voxel.png", auto_close=False)

    pl.open_movie(f"output-mov/{data_index}-voxel.mov")
    r = mov_r
    #The lower this value the higher quality the circle is with more points generated
    stepSize = mov_step
    t = 0
    while t < 2 * math.pi:
        pl.camera.position = (-0.3484873884891845 + r * math.cos(t), -3.5536092557145635 + r * math.sin(t), -5.746098824833765)
        pl.camera.focal_point = (0.148353260725683, 5.5250374342419875, 60.67324901052562)
        pl.camera.up = (-0.007880808650864862, -0.9907485765920729, 0.13548118258927996)
        pl.write_frame()
        t += stepSize

    pl.close()

def gt():
    print("Visualizing Ground Truth")
    voxel_size = 0.5
    min_mask = cloud_gt >= [-16,-31,0.0]
    max_mask = cloud_gt <= [16,1,32]
    min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
    max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
    filter_mask = min_mask & max_mask
    filtered_cloud = cloud_gt[filter_mask]

    xyz_q = np.floor(np.array(filtered_cloud/voxel_size)).astype(int) # quantized point values, here you will loose precision
    vox_grid = np.zeros((int(32/voxel_size), int(32/voxel_size), int(32/voxel_size))) #Empty voxel grid
    offsets = np.array([32, 62, 0])
    xyz_offset_q = xyz_q+offsets
    vox_grid[xyz_offset_q[:,0],xyz_offset_q[:,1],xyz_offset_q[:,2]] = 1 # Setting all voxels containitn a points equal to 1

    xyz_v = np.asarray(np.where(vox_grid == 1)) # get back indexes of populated voxels
    cloud_np = np.asarray([(pt-offsets)*voxel_size for pt in xyz_v.T])
        
    voxel_grid = create_voxel_grid(cloud_np)

    pl = pyvista.Plotter(off_screen=True)
    pl.set_background("white")
    pl.add_points(cloud_gt, scalars=points_rgb[mask.reshape(352000)], point_size=5, rgb=True)
    for idx, p in enumerate(cloud_np):
        pl.add_mesh(pyvista.Cube(p, voxel_size, voxel_size, voxel_size), show_edges=True, line_width=3, opacity=0.2, color="B10026")
    
    pl.show(cpos=[(-0.3484873884891845, -3.5536092557145635, -5.746098824833765),
        (0.148353260725683, 5.5250374342419875, 60.67324901052562),
        (-0.007880808650864862, -0.9907485765920729, 0.13548118258927996)],return_cpos=True, screenshot=f"output/{data_index}-gt.png", auto_close=False)
    
    pl.open_movie(f"output-mov/{data_index}-gt.mov")
    r = mov_r
    #The lower this value the higher quality the circle is with more points generated
    stepSize = mov_step
    t = 0
    while t < 2 * math.pi:
        pl.camera.position = (-0.3484873884891845 + r * math.cos(t), -3.5536092557145635 + r * math.sin(t), -5.746098824833765)
        pl.camera.focal_point = (0.148353260725683, 5.5250374342419875, 60.67324901052562)
        pl.camera.up = (-0.007880808650864862, -0.9907485765920729, 0.13548118258927996)
        pl.write_frame()
        t += stepSize

    pl.close()

def lac():
    print("Visualizing Lac")
    from models.lacGwcNet.networks.stackhourglass import PSMNet
    # load model
    affinity_settings = {}
    affinity_settings['win_w'] = 3
    affinity_settings['win_h'] = 3
    affinity_settings['dilation'] = [1, 2, 4, 8]

    model = PSMNet(maxdisp=192, struct_fea_c=4, fuse_mode="separate",
            affinity_settings=affinity_settings, udc=True, refine="csr").cuda()

    model = nn.DataParallel(model)
    model.eval()
    ckpt = torch.load("../models/lacGwcNet/checkpoint_9.tar")
    model.load_state_dict(ckpt["net"])

    model.eval()
    sample_left = torch.Tensor(left_img)
    sample_right = torch.Tensor(right_img)

    sample_left = torch.unsqueeze(sample_left, dim=0)
    sample_right = torch.unsqueeze(sample_right, dim=0)

    with torch.no_grad():
        disp_est_tn = model(sample_left.cuda(), sample_right.cuda(), None)[0]
        disp_est_np = tensor2numpy(disp_est_tn)
        disp_est = np.array(disp_est_np, dtype=np.float32)
        disp_est[disp_est < 0] = 0

    mask = disp_est > 0
    depth = f_u * baseline / (disp_est + 1. - mask)

    mask = disparity > 0
    depth_gt = f_u * baseline / (disparity + 1. - mask)

    mask = disp_est > 0
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = project_image_to_velo(points)

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

    voxel_size = 0.5
    min_mask = cloud >= [-16,-31,0.0]
    max_mask = cloud <= [16,1,32]
    min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
    max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
    filter_mask = min_mask & max_mask
    filtered_cloud = cloud[filter_mask]

    xyz_q = np.floor(np.array(filtered_cloud/voxel_size)).astype(int) # quantized point values, here you will loose precision
    vox_grid = np.zeros((int(32/voxel_size), int(32/voxel_size), int(32/voxel_size))) #Empty voxel grid
    offsets = np.array([32, 62, 0])
    xyz_offset_q = xyz_q+offsets
    vox_grid[xyz_offset_q[:,0],xyz_offset_q[:,1],xyz_offset_q[:,2]] = 1 # Setting all voxels containitn a points equal to 1

    xyz_v = np.asarray(np.where(vox_grid == 1)) # get back indexes of populated voxels
    cloud_np = np.asarray([(pt-offsets)*voxel_size for pt in xyz_v.T])


    voxel_grid = create_voxel_grid(cloud_np)

    pl = pyvista.Plotter(off_screen=True)
    pl.set_background("white")
    pl.add_points(cloud_gt, scalars=points_rgb[mask.reshape(352000)], point_size=5, rgb=True)
    for idx, p in enumerate(cloud_np):
        pl.add_mesh(pyvista.Cube(p, voxel_size, voxel_size, voxel_size), show_edges=True, line_width=3, opacity=0.2, color="B10026")
    pl.show(cpos=[(-0.3484873884891845, -3.5536092557145635, -5.746098824833765),
        (0.148353260725683, 5.5250374342419875, 60.67324901052562),
        (-0.007880808650864862, -0.9907485765920729, 0.13548118258927996)], screenshot=f"output/{data_index}-lac.png", auto_close=False)

    pl.open_movie(f"output-mov/{data_index}-lac.mov")
    r = mov_r
    #The lower this value the higher quality the circle is with more points generated
    stepSize = mov_step
    t = 0
    while t < 2 * math.pi:
        pl.camera.position = (-0.3484873884891845 + r * math.cos(t), -3.5536092557145635 + r * math.sin(t), -5.746098824833765)
        pl.camera.focal_point = (0.148353260725683, 5.5250374342419875, 60.67324901052562)
        pl.camera.up = (-0.007880808650864862, -0.9907485765920729, 0.13548118258927996)
        pl.write_frame()
        t += stepSize

    pl.close()

lac()    
sgbm()
voxel()
gt()

# concat four videos into one
clip_gt = VideoFileClip(f"output-mov/{data_index}-gt.mov")
clip_voxel = VideoFileClip(f"output-mov/{data_index}-voxel.mov")
clip_sgbm = VideoFileClip(f"output-mov/{data_index}-sgbm.mov")
clip_lac = VideoFileClip(f"output-mov/{data_index}-lac.mov")

clip_gt_text = TextClip("Grount Truth", font="Times-Roman", color="Black", bg_color="White",
                   fontsize=70).set_duration(clip_lac.duration)
clip_voxel_text = TextClip("Ours", font="Times-Roman", color="Black", bg_color="White",
                   fontsize=70, stroke_color="Black", stroke_width=3).set_duration(clip_lac.duration)
clip_sgbm_text = TextClip("SGBM", font="Times-Roman", color="Black", bg_color="White",
                   fontsize=70).set_duration(clip_lac.duration)
clip_lac_text = TextClip("Lac-GwcNet", font="Times-Roman", color="Black", bg_color="White",
                   fontsize=70).set_duration(clip_lac.duration)
            
vis_clip = clips_array([[clip_gt, clip_voxel],
                            [clip_gt_text, clip_voxel_text],
                            [clip_sgbm, clip_lac],
                            [clip_sgbm_text, clip_lac_text]], bg_color=(255,255,255))

vis_clip.write_videofile(f"output-mov/{data_index}-final.mp4")