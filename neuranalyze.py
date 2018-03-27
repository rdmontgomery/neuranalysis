# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:17:11 2018

@author: rickmontgomery

Most of the code is cloned from PyVoxelizer
https://github.com/p-hofmann/PyVoxelizer
"""

import argparse
from itertools import chain
import logging, logging.handlers
import math
import numpy as np
import os
import pandas as pd

from mesh import get_scale_and_shift, scale_and_shift_triangle
from meshreader import MeshReader
from voxelintersect.triangle import Triangle, t_c_intersection, INSIDE
from voxelintersect.triangle import vertexes_to_c_triangle, triangle_lib


class BoundaryBox(object):
    """
    @type minimum: list[int]
    @type maximum: list[int]
    """

    minimum = None
    maximum = None

    def get_center(self):
        assert self.minimum, "BoundaryBox not initialized"
        return [
            int((self.maximum[0] + self.minimum[0])/2.0),
            int((self.maximum[1] + self.minimum[1])/2.0),
            int((self.maximum[2] + self.minimum[2])/2.0)
            ]

    def from_triangle(self, triangle):
        """
        @type triangle: Triangle
        """
        self.minimum[0] = math.floor(triangle.min(0))
        self.minimum[1] = math.floor(triangle.min(1))
        self.minimum[2] = math.floor(triangle.min(2))

        self.maximum[0] = math.ceil(triangle.max(0))
        self.maximum[1] = math.ceil(triangle.max(1))
        self.maximum[2] = math.ceil(triangle.max(2))

    def from_vertexes(self, vertex_1, vertex_2, vertex_3):
        """
        @type vertex_1: (float, float, float)
        @type vertex_2: (float, float, float)
        @type vertex_3: (float, float, float)
        """
        if self.minimum is None:
            self.minimum = [0, 0, 0]
            self.maximum = [0, 0, 0]

            self.minimum[0] = math.floor(
                    min([vertex_1[0], vertex_2[0], vertex_3[0]]))
            self.minimum[1] = math.floor(
                    min([vertex_1[1], vertex_2[1], vertex_3[1]]))
            self.minimum[2] = math.floor(
                    min([vertex_1[2], vertex_2[2], vertex_3[2]]))

            self.maximum[0] = math.ceil(
                    max([vertex_1[0], vertex_2[0], vertex_3[0]]))
            self.maximum[1] = math.ceil(
                    max([vertex_1[1], vertex_2[1], vertex_3[1]]))
            self.maximum[2] = math.ceil(
                    max([vertex_1[2], vertex_2[2], vertex_3[2]]))
        else:
            self.minimum[0] = math.floor(min(
                    [vertex_1[0], vertex_2[0], vertex_3[0], self.minimum[0]]))
            self.minimum[1] = math.floor(min(
                    [vertex_1[1], vertex_2[1], vertex_3[1], self.minimum[1]]))
            self.minimum[2] = math.floor(min(
                    [vertex_1[2], vertex_2[2], vertex_3[2], self.minimum[2]]))

            self.maximum[0] = math.ceil(max(
                    [vertex_1[0], vertex_2[0], vertex_3[0], self.maximum[0]]))
            self.maximum[1] = math.ceil(max(
                    [vertex_1[1], vertex_2[1], vertex_3[1], self.maximum[1]]))
            self.maximum[2] = math.ceil(max(
                    [vertex_1[2], vertex_2[2], vertex_3[2], self.maximum[2]]))


n_range = {-1, 0, 1}


def get_intersecting_voxels_depth_first(vertex_1, vertex_2, vertex_3):
    """

    @type vertex_1: numpy.ndarray
    @type vertex_2: numpy.ndarray
    @type vertex_3: numpy.ndarray

    @rtype: list[(int, int, int)]
    """
    c_lib = triangle_lib
    result_positions = []
    tmp_triangle = None
    searched = set()
    stack = set()

    seed = (int(vertex_1[0]), int(vertex_1[1]), int(vertex_1[2]))
    for x in n_range:
        for y in n_range:
            for z in n_range:
                neighbour = (seed[0] + x, seed[1] + y, seed[2] + z)
                if neighbour not in searched:
                    stack.add(neighbour)

    tmp = np.array([0.0, 0.0, 0.0])
    tmp_vertex_1 = np.array([0.0, 0.0, 0.0])
    tmp_vertex_2 = np.array([0.0, 0.0, 0.0])
    tmp_vertex_3 = np.array([0.0, 0.0, 0.0])
    if not c_lib:
        tmp_triangle = Triangle()
        tmp_triangle.set(tmp_vertex_1, tmp_vertex_2, tmp_vertex_3)
    while len(stack) > 0:
        position = stack.pop()
        searched.add(position)
        tmp[0] = 0.5 + position[0]
        tmp[1] = 0.5 + position[1]
        tmp[2] = 0.5 + position[2]

        # move raster to origin, test assumed triangle in relation to origin
        np.subtract(vertex_1, tmp, tmp_vertex_1)
        np.subtract(vertex_2, tmp, tmp_vertex_2)
        np.subtract(vertex_3, tmp, tmp_vertex_3)

        try:
            if c_lib:
                is_inside = c_lib.t_c_intersection(vertexes_to_c_triangle(
                        tmp_vertex_1,
                        tmp_vertex_2,
                        tmp_vertex_3)
                ) == INSIDE
            else:
                is_inside = t_c_intersection(tmp_triangle) == INSIDE
        except Exception:
            c_lib = None
            tmp_triangle = Triangle()
            tmp_triangle.set(tmp_vertex_1, tmp_vertex_2, tmp_vertex_3)
            is_inside = t_c_intersection(tmp_triangle) == INSIDE

        if is_inside:
            result_positions.append(position)

            neighbours = set()
            if tmp_vertex_2[0] < 0:
                neighbours.add((position[0] - 1, position[1], position[2]))
                if tmp_vertex_3[0] > 0:
                    neighbours.add((position[0] + 1, position[1], position[2]))
            else:
                neighbours.add((position[0] + 1, position[1], position[2]))
                if tmp_vertex_3[0] < 0:
                    neighbours.add((position[0] - 1, position[1], position[2]))

            if tmp_vertex_2[1] < 0:
                neighbours.add((position[0], position[1] - 1, position[2]))
                if tmp_vertex_3[1] > 0:
                    neighbours.add((position[0], position[1] + 1, position[2]))
            else:
                neighbours.add((position[0], position[1] + 1, position[2]))
                if tmp_vertex_3[1] < 0:
                    neighbours.add((position[0], position[1] - 1, position[2]))

            if tmp_vertex_2[2] < 0:
                neighbours.add((position[0], position[1], position[2] - 1))
                if tmp_vertex_3[2] > 0:
                    neighbours.add((position[0], position[1], position[2] + 1))
            else:
                neighbours.add((position[0], position[1], position[2] + 1))
                if tmp_vertex_3[2] < 0:
                    neighbours.add((position[0], position[1], position[2] - 1))

            for neighbour in neighbours:
                if neighbour not in searched:
                    stack.add(neighbour)
    del searched, stack
    return result_positions


def voxelize(file_path, resolution):
    """

    @type file_path: str
    @type resolution: int
    @type progress_bar: any
    """
    mesh_reader = MeshReader()
    if file_path.endswith('.zip'):
        mesh_reader.read_archive(file_path)
    else:
        mesh_reader.read(file_path)
    if not mesh_reader.has_triangular_facets():
        raise NotImplementedError("Unsupported polygonal face elements. \
                                  Only triangular facets supported.")

    list_of_triangles = list(mesh_reader.get_facets())
    scale, shift, triangle_count = get_scale_and_shift(
            list_of_triangles,
            resolution)
    voxels = set()
    bounding_box = BoundaryBox()
    for triangle in list_of_triangles:
        (vertex_1, vertex_2, vertex_3) = scale_and_shift_triangle(
                triangle,
                scale,
                shift)
        bounding_box.from_vertexes(vertex_1, vertex_2, vertex_3)
        voxels.update(get_intersecting_voxels_depth_first(
                vertex_1,
                vertex_2,
                vertex_3))
    center = bounding_box.get_center()
    while len(voxels) > 0:
        (x, y, z) = voxels.pop()
        yield x-center[0], y-center[1], z-center[2]


def get_bbox(file):
    mesh_reader = MeshReader()
    mesh_reader.read(file)
    list_of_triangles = list(mesh_reader.get_facets())
    xs = np.array(list(chain.from_iterable(list_of_triangles)))[:,0]
    ys = np.array(list(chain.from_iterable(list_of_triangles)))[:,1]
    zs = np.array(list(chain.from_iterable(list_of_triangles)))[:,2]
    bbox = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))

    return bbox


def save_npy(file, folder_name, voxels, L):
    L_str = f'{L:0.2f}'.replace(".", "_")
    name = f'{file.split(".")[0]}_L-{L_str}'
    np.save(os.path.join('binaries', folder_name, name + '.npy'), voxels)


def setup_logger():
    """Log to file and to console errors and warnings"""
    os.makedirs('logs', exist_ok=True)

    log_formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s')

    log_file_name = 'errors.txt'

    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(
            'logs',
            log_file_name),
        maxBytes=8388608,
        backupCount=30)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    file_handler.doRollover()

    console_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.NOTSET,
        handlers=[
            console_handler,
            file_handler])


if __name__ == '__main__':
    setup_logger()
    parser = argparse.ArgumentParser(
            description='''Converts obj file to voxels.
                           Saves a numpy binary and a png.''')
    parser.add_argument(
            '--folder',
            help='folder with OBJ files')
    parser.add_argument(
            '--npy',
            action='store_true',
            help='Use --npy to export voxels as npy data')
    args = parser.parse_args()

    os.makedirs('boxCounts', exist_ok=True)
    folder_name = args.folder.split('/')[-3][0] + args.folder.split('/')[-2]
    os.makedirs(os.path.join('boxCounts', folder_name), exist_ok=True)
    os.makedirs(os.path.join('binaries', folder_name), exist_ok=True)

    files = [file for file in os.listdir(args.folder) if file.endswith('.obj')]

    master_df = pd.DataFrame(columns=['file', 'res', 'L', 'N'])
    for file in files:
        df = pd.DataFrame(columns=['file', 'res', 'L', 'N'])
        file_path = os.path.join(args.folder, file)
        bbox = get_bbox(file_path)

        for res in np.geomspace(10, bbox/1.3207, 100):
            L = bbox/res * 0.3786
            voxels = np.empty([0, 3])
            for x, y, z in voxelize(file_path, res):
                voxels = np.append(voxels, [[x, y, z]], axis=0)

            logging.info(
                    f'{file:20} | L={L:8.2f} | N={len(voxels):8.0f}')

            df.loc[len(df)] = {
                    'file': file.split('.')[0],
                    'res': res,
                    'L': L,
                    'N': len(voxels)}
            if args.npy:
                save_npy(file, folder_name, voxels, L)

        df.to_csv(os.path.join(
                'boxCounts', folder_name, f'{file.split(".")[0]}_boxCounts.csv'))
        master_df = master_df.append(df)

    master_df.to_csv(folder_name + '_boxCounts.csv', index=False)
