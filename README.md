# piroplasma

[![License MIT](https://img.shields.io/pypi/l/piroplasma.svg?color=green)](https://github.com/MontpellierRessourcesImagerie/piroplasma/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/piroplasma.svg?color=green)](https://pypi.org/project/piroplasma)
[![Python Version](https://img.shields.io/pypi/pyversions/piroplasma.svg?color=green)](https://python.org)
[![tests](https://github.com/MontpellierRessourcesImagerie/piroplasma/workflows/tests/badge.svg)](https://github.com/MontpellierRessourcesImagerie/piroplasma/actions)
[![codecov](https://codecov.io/gh/MontpellierRessourcesImagerie/piroplasma/branch/main/graph/badge.svg)](https://codecov.io/gh/MontpellierRessourcesImagerie/piroplasma)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/piroplasma)](https://napari-hub.org/plugins/piroplasma)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

Automatic parasitemia estimation and piroplasma stage detection in blood smears.

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template] (None).

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `piroplasma` via [pip]:

```
pip install git+https://github.com/MontpellierRessourcesImagerie/piroplasma.git
```

## Tiling full images into training data

The classifier is trained on tiles. `piroplasma.tiling.tile_dataset` cuts the
full resolution images of a blood smear into such tiles and redistributes the
YOLO annotations that were predicted on the full image to the tiles they
belong to.

Annotations that are cut by the border of a tile are not complete cells any
more: their class is changed to the `border` class (3 by default), so that
they are not used as misleading training examples.

```python
from piroplasma.tiling import tile_dataset

tile_dataset(
    input_dir="/home/baecker/Documents/mri/in/2026/2247_bird_blood_smear/full_images",
    output_dir="/home/baecker/Documents/mri/in/2026/2247_bird_blood_smear/tiles",
    tile_size=1600,   # size of the (square) tiles
    overlap=1 / 3,    # overlap between neighbouring tiles
    border_class=3,   # class of the cells that touch the border of a tile
)
```

For every image `<name>.tif` of `input_dir` the annotation file `<name>.txt`
(YOLO format: `class x_center y_center width height`, normalized coordinates)
is expected. The tiles and their annotation files are written to `output_dir`
as `<name>_tile_<index>.tif` and `<name>_tile_<index>.txt`.

The tiles are cut exactly like the classifier does it (the layout of
`ImageTiler2D` of [microglia-analyzer](https://github.com/MontpellierRessourcesImagerie/microglia-analyzer)),
so that the tiles can be fed to the model directly.

Useful parameters:

| Parameter | Description |
|-----------|-------------|
| `tile_size` | size of the tiles in pixels (default `1600`) |
| `overlap` | overlap between neighbouring tiles, as a fraction of `tile_size` (default `0.33`) |
| `border_class` | class of the annotations that touch the border of a tile (default `3`) |
| `border_margin` | additional margin (in pixels) inside the tile: annotations closer than that to the edge of a tile are considered as touching the border as well (default `0`) |
| `overwrite` | if `False` (default), tiles that already exist are not written again, which allows to interrupt and resume the tiling of a large dataset |

### The classes behind it

`tile_dataset` is a thin convenience wrapper around `TilingDataset`, which does
the actual work. The module is built around small objects that can also be
used on their own:

| Class | Role |
|-------|------|
| `BoundingBox` | one annotation in absolute pixel coordinates (`from_yolo`, `to_yolo`, `relabeled`, `translated`) |
| `TileRegion` | the footprint of one tile in the full image (`intersection`, `contains`) |
| `AnnotationSet` | a collection of boxes, read from and written to a YOLO file |
| `ImageIO` | reading (memory mapped when possible) and writing of the image files |
| `TileLayout` | computes the regions an image is cut into, with the layout of the classifier |
| `TilingOptions` | the parameters of a tiling job (tile size, overlap, border class, ...) |
| `TiledImage` | one image and its annotations, cut into tiles (`annotations_for`, `write_tiles`) |
| `TilingDataset` | a folder of images (`image_files`, `run`) |

```python
from piroplasma.tiling import TilingDataset

dataset = TilingDataset(input_dir=".../full_images", output_dir=".../tiles")
written = dataset.run()   # {tile image path: annotation path}
```



## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"piroplasma" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/MontpellierRessourcesImagerie/piroplasma/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
