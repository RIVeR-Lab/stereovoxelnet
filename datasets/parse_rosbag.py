import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Header
from PIL import Image
import argparse
import rosbag
import os

parser = argparse.ArgumentParser(description='MobileStereoNet')
parser.add_argument('--bag_file', type=str, required=True,
                    help='Bag file to be parsed')
parser.add_argument('--skip_num', type=int, default=2,
                    help='keep 1/N fraction of data')
parser.add_argument('--dataset_mode', type=str, default="train",
                    help='Training set or Testing set')
args = parser.parse_args()

topics = ["/zed2/left/image_rect_color","/zed2/right/image_rect_color","/rslidar_points"]

def filter_cloud(cloud):
    min_mask = cloud >= [-3.2,-6.3,0.0]
    max_mask = cloud <= [3.2,0.1,6.4]
    min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
    max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
    filter_mask = min_mask & max_mask
    filtered_cloud = cloud[filter_mask]

    # filter by camera view
    c_u = 447.59914779663086
    c_v = 255.83612823486328
    f_u = 365.68
    f_v = 365.68
    baseline = 0.12
    image_shape = (880, 495)

    uv_depth = np.zeros((len(filtered_cloud), 2))
    uv_depth[:,0] = (filtered_cloud[:,0]*f_u)/filtered_cloud[:,2] + c_u
    uv_depth[:,1] = (filtered_cloud[:,1]*f_v)/filtered_cloud[:,2] + c_v

    min_mask = uv_depth >= [0,0]
    max_mask = uv_depth <= [880,495]
    min_mask = min_mask[:, 0] & min_mask[:, 1] 
    max_mask = max_mask[:, 0] & max_mask[:, 1]
    filter_mask = min_mask & max_mask
    filtered_cloud = filtered_cloud[filter_mask]

    return filtered_cloud

def calc_voxel_grid(filtered_cloud, voxel_size):
    xyz_q = np.floor(np.array(filtered_cloud/voxel_size)).astype(int) # quantized point values, here you will loose precision
    vox_grid = np.zeros((int(6.4/voxel_size), int(6.4/voxel_size), int(6.4/voxel_size))) #Empty voxel grid
    offsets = np.array([32, 63, 0])
    xyz_offset_q = xyz_q+offsets
    vox_grid[xyz_offset_q[:,0],xyz_offset_q[:,1],xyz_offset_q[:,2]] = 1 # Setting all voxels containitn a points equal to 1

    xyz_v = np.asarray(np.where(vox_grid == 1)) # get back indexes of populated voxels
    cloud_np = np.asarray([(pt-offsets)*voxel_size for pt in xyz_v.T])
    return vox_grid, cloud_np

class Bag:
    def listen_image(self, data:Image, side):
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[:-95,:,:]

        if side == "left":
            self.image_pair[0] = frame
        elif side == "right":
            self.image_pair[1] = frame
        else:
            raise NotImplementedError()

        if self.image_pair[0] is not None and self.image_pair[1] is not None:
            # skip a few
            self.skip_counter -= 1
            if self.skip_counter != 0:
                self.image_pair = [None, None]
                return
            self.skip_counter = args.skip_num

            # save image and voxel grid
            # im_L = Image.fromarray(self.image_pair[0])
            # im_L.save(f"ISEC/ISEC5/left/{self.counter}.png")
            # im_R = Image.fromarray(self.image_pair[1])
            # im_R.save(f"ISEC/ISEC5/right/{self.counter}.png")
            np.save(f"ISEC/{args.dataset_mode}/left/{args.bag_file}_{self.counter}.npy", self.image_pair[0])
            np.save(f"ISEC/{args.dataset_mode}/right/{args.bag_file}_{self.counter}.npy", self.image_pair[1])
            np.save(f"ISEC/{args.dataset_mode}/voxel/{args.bag_file}_{self.counter}.npy", self.rslidar_voxels)
            self.counter += 1
            print(f"finished processing {args.bag_file}_{self.counter}.npy data")

            self.image_pair = [None, None]
    def listen_pc(self, data:PointCloud2):
        # transform into ZED frame
        if self.mat44 is None:
            self.mat44 = self.listener.asMatrix("zed_left", data.header)
        def xf(p):
            xyz = tuple(np.dot(self.mat44, np.array([p[0], p[1], p[2], 1.0])))[:3]
            return xyz
        point_list = [xf(p) for p in pc2.read_points(data)]
        point_np = np.array(point_list)

        # gt_pcd = o3d.geometry.PointCloud()
        # gt_pcd.points = o3d.utility.Vector3dVector(point_np)

        filtered_cloud = filter_cloud(point_np)
        vox_grid, cloud_np = calc_voxel_grid(filtered_cloud, 0.1)
        
        # self.rslidar_points = point_np
        self.rslidar_voxels = vox_grid
    
    def pub_pc(self, cloud_np, time):
        points_rgb = np.ones((len(cloud_np), 1))
        color_pl = points_rgb[:, 0] * 65536 * 255
        color_pl = np.expand_dims(color_pl, axis=-1)
        color_pl = color_pl.astype(np.uint32)

        # concat to ROS pointcloud foramt
        concat_pl = np.concatenate((cloud_np, color_pl), axis=1)
        points = concat_pl.tolist()

        # TODO: needs to fix this type conversion
        for i in range(len(points)):
            points[i][3] = int(points[i][3])

        header = Header()
        header.stamp = time
        header.frame_id = "zed_left"

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.UINT32, 1),
                ]
        pc = pc2.create_cloud(header, fields, points)

        self.point_cloud_pub.publish(pc)

    def __init__(self) -> None:
        self.rslidar_points = None
        self.rslidar_voxels = None
        self.image_pair = [None, None]

        self.point_cloud_pub = rospy.Publisher(
            "/voxels", PointCloud2, queue_size=1)
        self.mat44 = np.load("mat44.npy")
        self.bridge = CvBridge()
        self.counter = 0
        self.skip_counter = args.skip_num

if __name__ == "__main__":
    os.makedirs(f"ISEC/{args.dataset_mode}/left/", exist_ok=True)
    os.makedirs(f"ISEC/{args.dataset_mode}/right/", exist_ok=True)
    os.makedirs(f"ISEC/{args.dataset_mode}/voxel/", exist_ok=True)

    rospy.init_node(f"{args.bag_file.split('.')[0]}_parse_bag")
    isec = rosbag.Bag(args.bag_file)

    bag = Bag()

    for topic, msg, t in isec.read_messages(topics=topics):
        if topic == "/zed2/left/image_rect_color":
            bag.listen_image(msg, "left")
        elif topic == "/zed2/right/image_rect_color":
            bag.listen_image(msg, "right")
        elif topic == "/rslidar_points":
            bag.listen_pc(msg)