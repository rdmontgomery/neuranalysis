# neuranalysis
Long live Taylor lab.

This code is an instantiation of [PyVoxelizer](https://github.com/p-hofmann/PyVoxelizer).
From a command line terminal, type
```
python neuranalyze.py --obj 1_0001Full100.obj --res 50
```

The program outputs 3 things.

1. It prints the number of voxels necessary to represent the obj file at the given resolution.
2. It produces a .npy binary file with the voxel information. This file can be loaded using neuranalysis.ipynb.
3. It produces a 3D scatter plot image.
