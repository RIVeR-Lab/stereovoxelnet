from .dataset import VoxelDataset, VoxelKITTIDataset, VoxelDSDataset, VoxelISECDataset

__datasets__ = {
    "voxel": VoxelDataset,
    "voxelkitti": VoxelKITTIDataset,
    "voxelds": VoxelDSDataset,
    "voxelisec": VoxelISECDataset
}
