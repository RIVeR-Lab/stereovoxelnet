# Copyright (c) 2021. All rights reserved.
from .Voxel2D import Voxel2D
from .Voxel2D_sparse import Voxel2D as Voxel2DSparse
from .Voxel2D_hie import Voxel2D as Voxel2DHie
from .submodule import model_loss, calc_IoU

__models__ = {
    "Voxel2D": Voxel2D,
    "Voxel2D_sparse": Voxel2DSparse,
    "Voxel2D_hie": Voxel2DHie,
}
