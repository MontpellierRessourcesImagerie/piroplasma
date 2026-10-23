"""Tiling of full resolution images and redistribution of YOLO annotations.

The classifier is trained on tiles of a fixed size. This module cuts the
"full images" (whole slide images of a blood smear) into such tiles and
redistributes the YOLO annotations that were predicted on the full image to
the tiles they belong to.

Annotations that are cut by the edge of a tile (or that lie within
``border_margin`` pixels of a tile edge) are not complete cells any more.
Their class is therefore changed to the border class, so that they are not
used as (misleading) training examples for the cell classes.

The tiling layout is computed with :class:`ImageTiler2D` of
``microglia_analyzer`` - the very same class that is used by
:class:`piroplasma.inference.PiroplasmaInference`. This guarantees that the
tiles produced here are exactly the tiles the classifier works on.

The main entry point is :class:`TilingDataset`, or the convenience function
:func:`tile_dataset` that wraps it.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import tifffile
from microglia_analyzer.tiles.tiler import ImageTiler2D

try:  # imageio is only needed for the non TIFF formats
    import imageio.v3 as iio
except ImportError:  # pragma: no cover
    iio = None


#: Image file extensions that are looked for in the input folder.
IMAGE_EXTENSIONS: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


@dataclasses.dataclass(frozen=True)
class BoundingBox:
    """An annotation in absolute pixel coordinates.

    The box is half open, i.e. it covers the pixels
    ``[x_min, x_max) x [y_min, y_max)``.
    """

    label: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @classmethod
    def from_yolo(
        cls,
        label: int,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        image_width: int,
        image_height: int,
    ) -> "BoundingBox":
        """Build a box from normalized YOLO values.

        Parameters
        ----------
        label: int
            Class id of the box.
        x_center, y_center, width, height: float
            Normalized YOLO values of the box.
        image_width, image_height: int
            Size of the image the box belongs to.

        Returns
        -------
        BoundingBox
            The box in absolute pixel coordinates, clipped to the image.
        """
        x_min = int(round((x_center - width / 2) * image_width))
        x_max = int(round((x_center + width / 2) * image_width))
        y_min = int(round((y_center - height / 2) * image_height))
        y_max = int(round((y_center + height / 2) * image_height))
        # clip to the image
        x_min = max(0, min(x_min, image_width))
        x_max = max(0, min(x_max, image_width))
        y_min = max(0, min(y_min, image_height))
        y_max = max(0, min(y_max, image_height))
        return cls(label, x_min, y_min, x_max, y_max)

    def to_yolo(self, image_width: int, image_height: int) -> str:
        """Format the box as a YOLO annotation line.

        Parameters
        ----------
        image_width, image_height: int
            Size of the image (or tile) the box belongs to.

        Returns
        -------
        str
            The annotation line ``"class x_center y_center width height"``.
        """
        x_center = ((self.x_min + self.x_max) / 2) / image_width
        y_center = ((self.y_min + self.y_max) / 2) / image_height
        width = (self.x_max - self.x_min) / image_width
        height = (self.y_max - self.y_min) / image_height
        return f"{self.label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    @property
    def is_valid(self) -> bool:
        """True if the box covers at least one pixel."""
        return self.x_min < self.x_max and self.y_min < self.y_max

    def relabeled(self, label: int) -> "BoundingBox":
        """Return a copy of the box with another class id."""
        return dataclasses.replace(self, label=label)

    def translated(self, dx: int, dy: int) -> "BoundingBox":
        """Return a copy of the box, moved by ``dx`` pixels horizontally and ``dy`` vertically."""
        return dataclasses.replace(
            self,
            x_min=self.x_min + dx,
            y_min=self.y_min + dy,
            x_max=self.x_max + dx,
            y_max=self.y_max + dy,
        )


@dataclasses.dataclass(frozen=True)
class TileRegion:
    """A rectangular region of an image: the footprint of one tile."""

    y_min: int
    x_min: int
    y_max: int
    x_max: int

    @property
    def height(self) -> int:
        """Height of the region in pixels."""
        return self.y_max - self.y_min

    @property
    def width(self) -> int:
        """Width of the region in pixels."""
        return self.x_max - self.x_min

    def intersection(self, box: BoundingBox) -> Optional[BoundingBox]:
        """Return the part of the box that is inside the region, if any.

        Parameters
        ----------
        box: BoundingBox
            The box, in the coordinate system of the full image.

        Returns
        -------
        BoundingBox or None
            The intersection of the box with the region, in the coordinate
            system of the full image, or None if the box does not intersect
            the region.
        """
        x_min = max(box.x_min, self.x_min)
        y_min = max(box.y_min, self.y_min)
        x_max = min(box.x_max, self.x_max)
        y_max = min(box.y_max, self.y_max)
        if x_min >= x_max or y_min >= y_max:
            return None
        return BoundingBox(box.label, x_min, y_min, x_max, y_max)

    def contains(self, box: BoundingBox, margin: int = 0) -> bool:
        """Tell whether the box lies completely inside the region.

        Parameters
        ----------
        box: BoundingBox
            The box, in the coordinate system of the full image.
        margin: int, default 0
            The box must be at least that many pixels away from the edges of
            the region.

        Returns
        -------
        bool
            True if the box is inside the region and not too close to its
            edges, i.e. if the box is not cut by the border of the tile.
        """
        return (
            box.x_min >= self.x_min + margin
            and box.y_min >= self.y_min + margin
            and box.x_max <= self.x_max - margin
            and box.y_max <= self.y_max - margin
        )


class AnnotationSet:
    """A collection of bounding boxes, that can be read from and written to
    a YOLO annotation file.

    The YOLO format is ``class x_center y_center width height``, with the last
    four values normalized to the size of the image. Additional columns (e.g. a
    confidence score) are ignored when reading.
    """

    def __init__(self, boxes: Sequence[BoundingBox] = ()) -> None:
        self.boxes: List[BoundingBox] = list(boxes)

    def __len__(self) -> int:
        return len(self.boxes)

    def __iter__(self) -> Iterator[BoundingBox]:
        return iter(self.boxes)

    def __bool__(self) -> bool:
        return bool(self.boxes)

    def __getitem__(self, index: int) -> BoundingBox:
        return self.boxes[index]

    def add(self, box: BoundingBox) -> None:
        """Add a box to the set."""
        self.boxes.append(box)

    def extend(self, boxes: Sequence[BoundingBox]) -> None:
        """Add several boxes to the set."""
        self.boxes.extend(boxes)

    @classmethod
    def from_yolo_file(
        cls, path: pathlib.Path, image_width: int, image_height: int
    ) -> "AnnotationSet":
        """Read a YOLO annotation file.

        Parameters
        ----------
        path: pathlib.Path
            Path of the annotation file.
        image_width, image_height: int
            Size of the image the annotations belong to.

        Returns
        -------
        AnnotationSet
            The boxes of the file, in absolute pixel coordinates. Empty and
            malformed lines are skipped.
        """
        boxes: List[BoundingBox] = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # skip empty or malformed lines
                label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                box = BoundingBox.from_yolo(
                    label, x_center, y_center, width, height, image_width, image_height
                )
                if box.is_valid:
                    boxes.append(box)
        return cls(boxes)

    def to_yolo_file(
        self, path: pathlib.Path, image_width: int, image_height: int
    ) -> None:
        """Write the boxes to a YOLO annotation file.

        Parameters
        ----------
        path: pathlib.Path
            Destination path.
        image_width, image_height: int
            Size of the image (or tile) the boxes belong to.
        """
        with open(path, "w") as f:
            for box in self.boxes:
                f.write(box.to_yolo(image_width, image_height) + "\n")


class ImageIO:
    """Reading and writing of the image files."""

    @staticmethod
    def read(path: pathlib.Path) -> np.ndarray:
        """Open an image and return it as a NumPy array.

        Uncompressed TIFF files - which is what the full resolution images are
        - are memory mapped, so that only the parts that are actually needed
        are read from disk. Everything else is read into memory.

        Parameters
        ----------
        path: pathlib.Path
            Path of the image.

        Returns
        -------
        numpy.ndarray
            The image, with shape ``(height, width)`` or
            ``(height, width, channels)``.
        """
        suffix = path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            try:
                return tifffile.memmap(path)
            except Exception:  # not memory mappable (e.g. compressed) -> read it
                return tifffile.imread(path)
        if iio is None:  # pragma: no cover
            raise ImportError("imageio is required to read " + str(path))
        return iio.imread(path)

    @staticmethod
    def write(path: pathlib.Path, image: np.ndarray) -> None:
        """Write an image to disk.

        Parameters
        ----------
        path: pathlib.Path
            Destination path.
        image: numpy.ndarray
            The image to write.
        """
        if path.suffix.lower() in (".tif", ".tiff"):
            tifffile.imwrite(path, np.asarray(image))
            return
        if iio is None:  # pragma: no cover
            raise ImportError("imageio is required to write " + str(path))
        iio.imwrite(path, np.asarray(image))


class TileLayout:
    """The layout of the tiles an image is cut into.

    The layout is the one of :class:`ImageTiler2D`, i.e. the same layout the
    classifier uses.
    """

    def __init__(self, tile_size: int = 1600, overlap: float = 0.33) -> None:
        """
        Parameters
        ----------
        tile_size: int, default 1600
            Size of the (square) tiles in pixels.
        overlap: float, default 0.33
            Overlap between two neighbouring tiles, as a fraction of
            ``tile_size``. ``0`` means no overlap.
        """
        self.tile_size = tile_size
        self.overlap = overlap

    def regions(self, shape: Tuple[int, ...]) -> List[TileRegion]:
        """Compute the regions (tiles) an image is cut into.

        If the image is smaller than one tile, the whole image is used as a
        single tile.

        Parameters
        ----------
        shape: tuple
            Shape of the image ``(height, width, ...)``.

        Returns
        -------
        list of TileRegion
            The regions, in absolute pixel coordinates.
        """
        height, width = shape[:2]
        if height < self.tile_size or width < self.tile_size:
            # The image is too small to be cut: use it as a single tile.
            return [TileRegion(0, 0, height, width)]
        tiler = ImageTiler2D(self.tile_size, int(round(self.tile_size * self.overlap)), shape)
        return [
            TileRegion(
                patch.ul_corner[0], patch.ul_corner[1], patch.lr_corner[0], patch.lr_corner[1]
            )
            for patch in tiler.layout
        ]


@dataclasses.dataclass(frozen=True)
class TilingOptions:
    """The parameters of a tiling job."""

    tile_size: int = 1600
    overlap: float = 0.33
    border_class: int = 3
    border_margin: int = 0
    overwrite: bool = False
    min_box_size: int = 0


class TiledImage:
    """One full resolution image and its annotations.

    The image is cut into tiles, and the annotations are distributed to the
    tiles they belong to. Annotations that are cut by the border of a tile get
    the border class, because such a cell is not completely visible in that
    tile.
    """

    def __init__(
        self,
        image_path: pathlib.Path,
        annotation_path: pathlib.Path,
        output_dir: pathlib.Path,
        options: Optional[TilingOptions] = None,
    ) -> None:
        """
        Parameters
        ----------
        image_path: pathlib.Path
            Path of the full resolution image.
        annotation_path: pathlib.Path
            Path of the YOLO annotation file belonging to the image.
        output_dir: pathlib.Path
            Folder where the tiles and their annotations are written.
        options: TilingOptions, optional
            The parameters of the tiling job.
        """
        self.image_path = pathlib.Path(image_path)
        self.annotation_path = pathlib.Path(annotation_path)
        self.output_dir = pathlib.Path(output_dir)
        self.options = options if options is not None else TilingOptions()
        self._image: Optional[np.ndarray] = None
        self._annotations: Optional[AnnotationSet] = None

    @property
    def image(self) -> np.ndarray:
        """The full resolution image (memory mapped when possible)."""
        if self._image is None:
            self._image = ImageIO.read(self.image_path)
        return self._image

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the full resolution image."""
        return self.image.shape

    @property
    def width(self) -> int:
        """Width of the full resolution image."""
        return self.shape[1]

    @property
    def height(self) -> int:
        """Height of the full resolution image."""
        return self.shape[0]

    @property
    def annotations(self) -> AnnotationSet:
        """The annotations of the full resolution image, in pixel coordinates."""
        if self._annotations is None:
            self._annotations = AnnotationSet.from_yolo_file(
                self.annotation_path, self.width, self.height
            )
        return self._annotations

    def regions(self) -> List[TileRegion]:
        """The regions the image is cut into."""
        layout = TileLayout(self.options.tile_size, self.options.overlap)
        return layout.regions(self.shape)

    def annotations_for(self, region: TileRegion) -> AnnotationSet:
        """Distribute the annotations to one tile.

        The annotations that are cut by the border of the tile (or that are
        closer than ``border_margin`` pixels to it) get the border class. The
        returned boxes are in the coordinate system of the tile.

        Parameters
        ----------
        region: TileRegion
            The region of the tile, in the coordinate system of the full image.

        Returns
        -------
        AnnotationSet
            The annotations of the tile.
        """
        boxes: List[BoundingBox] = []
        # Track which boxes received the border class *because* of this tile
        # (rather than already having it in the source annotations).
        was_cut: List[bool] = []
        for box in self.annotations:
            intersection = region.intersection(box)
            if intersection is None:
                continue  # the box is not in this tile
            cut = False
            if not region.contains(box, self.options.border_margin):
                # the box is cut by the border of the tile (or too close to it)
                intersection = intersection.relabeled(self.options.border_class)
                cut = box.label != self.options.border_class
            boxes.append(
                intersection.translated(dx=-region.x_min, dy=-region.y_min)
            )
            was_cut.append(cut)
        # Drop the fragments of cut boxes that became smaller than min_box_size.
        # Boxes that were small to begin with are kept: only the pieces created
        # by cutting a box at the tile border are removed.
        if getattr(self.options, "min_box_size", 0) > 0:
            min_sz = self.options.min_box_size
            boxes = [
                box
                for box, cut in zip(boxes, was_cut)
                if not cut
                or (box.x_max - box.x_min) >= min_sz
                and (box.y_max - box.y_min) >= min_sz
            ]
        return AnnotationSet(boxes)

    def tile_paths(self, index: int) -> Tuple[pathlib.Path, pathlib.Path]:
        """Return the paths of the tile image and of its annotation file.

        Parameters
        ----------
        index: int
            Index of the tile.

        Returns
        -------
        tuple
            The path of the tile image and the path of its annotation file.
        """
        stem = f"{self.image_path.stem}_tile_{index:03d}"
        return (
            self.output_dir / f"{stem}{self.image_path.suffix}",
            self.output_dir / f"{stem}.txt",
        )

    def write_tiles(self) -> Dict[str, str]:
        """Cut the image into tiles and write them with their annotations.

        Returns
        -------
        dict
            Mapping ``tile image path -> annotation path`` of the written
            tiles.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        written: Dict[str, str] = {}
        for index, region in enumerate(self.regions()):
            tile_path, annotation_path = self.tile_paths(index)
            if (
                tile_path.exists()
                and annotation_path.exists()
                and not self.options.overwrite
            ):
                written[str(tile_path)] = str(annotation_path)
                continue
            tile = np.asarray(self.image[region.y_min : region.y_max, region.x_min : region.x_max])
            annotations = self.annotations_for(region)
            ImageIO.write(tile_path, tile)
            annotations.to_yolo_file(annotation_path, region.width, region.height)
            written[str(tile_path)] = str(annotation_path)
        return written


class TilingDataset:
    """Cuts a folder of full resolution images into tiles and adapts their
    annotations.

    For every image of ``input_dir`` a YOLO annotation file with the same
    basename is expected (``<image>.txt``). Images without annotation file are
    skipped. The tiles and their annotation files are written to ``output_dir``
    with the names ``<image>_tile_<index>.<ext>`` and ``<image>_tile_<index>.txt``.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        tile_size: int = 1600,
        overlap: float = 0.33,
        border_class: int = 3,
        border_margin: int = 0,
        image_extensions: Sequence[str] = IMAGE_EXTENSIONS,
        overwrite: bool = False,
        min_box_size: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        input_dir: str
            Folder with the full resolution images and their YOLO annotations.
        output_dir: str
            Folder where the tiles and the new annotation files are written. It
            is created if it does not exist.
        tile_size: int, default 1600
            Size of the (square) tiles in pixels.
        overlap: float, default 0.33
            Overlap between two neighbouring tiles, as a fraction of
            ``tile_size``. ``0`` means no overlap.
        border_class: int, default 3
            Class id that is assigned to the annotations that touch the border
            of a tile.
        border_margin: int, default 0
            Additional margin (in pixels) inside the tile: annotations that are
            closer than that to the edge of a tile are considered as touching
            the border as well.
        image_extensions: sequence of str
            The image file extensions that are looked for.
        overwrite: bool, default False
            If False, tiles that already exist are not written again. This
            allows to interrupt and resume the tiling of a large dataset.
        min_box_size: int, default 0
            Minimum width and height (in pixels) of a box. Fragments of boxes
            that were cut by the border of a tile - i.e. that received the
            border class because of the tiling - and that are smaller than this
            are dropped. Boxes that were already smaller in the source
            annotations are kept. ``0`` disables the filtering.
        """
        self.input_dir = pathlib.Path(input_dir).expanduser().resolve()
        self.output_dir = pathlib.Path(output_dir).expanduser().resolve()
        self.image_extensions = tuple(e.lower() for e in image_extensions)
        self.options = TilingOptions(
            tile_size=tile_size,
            overlap=overlap,
            border_class=border_class,
            border_margin=border_margin,
            overwrite=overwrite,
            min_box_size=min_box_size,
        )
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"{self.input_dir} is not a directory")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def image_files(self) -> List[pathlib.Path]:
        """The image files of the input folder, sorted by name."""
        return sorted(
            p
            for p in self.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.image_extensions
        )

    def tiled_image(self, image_path: pathlib.Path) -> TiledImage:
        """Build the :class:`TiledImage` of one image of the input folder."""
        return TiledImage(
            image_path,
            image_path.with_suffix(".txt"),
            self.output_dir,
            self.options,
        )

    def run(self) -> Dict[str, str]:
        """Tile every image of the input folder.

        Returns
        -------
        dict
            Mapping ``tile image path -> annotation path`` of all the tiles.
        """
        written: Dict[str, str] = {}
        for image_path in self.image_files():
            annotation_path = image_path.with_suffix(".txt")
            if not annotation_path.is_file():
                print(f"[tiling] No annotation file for {image_path.name} - skipped.")
                continue
            written.update(self.tiled_image(image_path).write_tiles())
        return written


def tile_dataset(
    input_dir: str,
    output_dir: str,
    tile_size: int = 1600,
    overlap: float = 0.33,
    border_class: int = 3,
    border_margin: int = 0,
    image_extensions: Sequence[str] = IMAGE_EXTENSIONS,
    overwrite: bool = False,
    min_box_size: int = 0,
) -> Dict[str, str]:
    """Cut a folder of full resolution images into tiles and adapt the annotations.

    Annotations are given in the YOLO format ``class x_center y_center width
    height`` with normalized coordinates and are re-distributed to the tiles
    they belong to. Every annotation that is cut by the border of a tile gets
    the border class, because such a cell is not completely visible in that
    tile.

    Parameters
    ----------
    input_dir: str
        Folder with the full resolution images and their YOLO annotations.
    output_dir: str
        Folder where the tiles and the new annotation files are written. It is
        created if it does not exist.
    tile_size: int, default 1600
        Size of the (square) tiles in pixels.
    overlap: float, default 0.33
        Overlap between two neighbouring tiles, as a fraction of ``tile_size``.
        ``0`` means no overlap.
    border_class: int, default 3
        Class id that is assigned to the annotations that touch the border of a
        tile.
    border_margin: int, default 0
        Additional margin (in pixels) inside the tile: annotations that are
        closer than that to the edge of a tile are considered as touching the
        border as well.
    image_extensions: sequence of str
        The image file extensions that are looked for.
    overwrite: bool, default False
        If False, tiles that already exist are not written again. This allows
        to interrupt and resume the tiling of a large dataset.
    min_box_size: int, default 0
        Minimum width and height (in pixels) of a box. Fragments of boxes that
        were cut by the border of a tile - i.e. that received the border class
        because of the tiling - and that are smaller than this are dropped.
        Boxes that were already smaller in the source annotations are kept.
        ``0`` disables the filtering.

    Returns
    -------
    dict
        Mapping ``tile image path -> annotation path`` of all the tiles.

    Examples
    --------
    >>> from piroplasma.tiling import tile_dataset
    >>> tile_dataset(
    ...     input_dir="/path/to/full_images",
    ...     output_dir="/path/to/tiles",
    ...     tile_size=1600,
    ...     overlap=0.33,
    ...     border_class=3,
    ... )
    """
    return TilingDataset(
        input_dir=input_dir,
        output_dir=output_dir,
        tile_size=tile_size,
        overlap=overlap,
        border_class=border_class,
        border_margin=border_margin,
        image_extensions=image_extensions,
        overwrite=overwrite,
        min_box_size=min_box_size,
    ).run()
