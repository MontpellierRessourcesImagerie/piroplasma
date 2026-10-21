import numpy as np
import pytest
import tifffile

from piroplasma.tiling import (
    AnnotationSet,
    BoundingBox,
    TileRegion,
    TiledImage,
    TilingDataset,
    TilingOptions,
    tile_dataset,
)


def yolo_lines(boxes, width, height):
    """Build the content of a YOLO file from boxes in pixel coordinates."""
    lines = []
    for label, x_min, y_min, x_max, y_max in boxes:
        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height
        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height
        lines.append(f"{label} {x_center} {y_center} {box_width} {box_height}")
    return "\n".join(lines) + "\n"


def make_tiled_image(tmp_path, boxes=(), shape=(200, 200, 3), **options):
    """Create a TiledImage with a dummy image and a YOLO annotation file."""
    image_path = tmp_path / "slide.tif"
    tifffile.imwrite(image_path, np.zeros(shape, dtype=np.uint8))
    annotation_path = image_path.with_suffix(".txt")
    height, width = shape[:2]
    annotation_path.write_text(yolo_lines(boxes, width, height))
    return TiledImage(
        image_path,
        annotation_path,
        tmp_path / "out",
        TilingOptions(tile_size=100, overlap=0.0, **options),
    )


def test_annotation_set_from_yolo_file(tmp_path):
    txt_path = tmp_path / "image.txt"
    txt_path.write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "\n"
        "1 0.25 0.75 0.1 0.1 0.87\n"  # extra column (confidence) is ignored
        "malformed line\n"
    )
    annotations = AnnotationSet.from_yolo_file(txt_path, image_width=200, image_height=100)
    assert len(annotations) == 2
    assert annotations[0] == BoundingBox(0, 80, 40, 120, 60)
    assert annotations[1] == BoundingBox(1, 40, 70, 60, 80)


def test_bounding_box_yolo_roundtrip():
    box = BoundingBox.from_yolo(1, 0.5, 0.5, 0.2, 0.4, image_width=200, image_height=100)
    assert box == BoundingBox(1, 80, 30, 120, 70)
    assert box.to_yolo(200, 100) == "1 0.500000 0.500000 0.200000 0.400000"
    assert box.relabeled(3).label == 3
    assert box.translated(dx=-80, dy=-30) == BoundingBox(1, 0, 0, 40, 40)


def test_tile_region_intersection_and_contains():
    region = TileRegion(y_min=0, x_min=100, y_max=100, x_max=200)
    assert region.width == 100 and region.height == 100
    inside = BoundingBox(1, 110, 10, 150, 50)
    assert region.intersection(inside) == inside
    assert region.contains(inside)
    outside = BoundingBox(0, 90, 10, 130, 50)
    assert region.intersection(outside) == BoundingBox(0, 100, 10, 130, 50)
    assert not region.contains(outside)
    assert region.intersection(BoundingBox(0, 0, 0, 50, 50)) is None


def test_annotations_for_tile_relabels_cut_boxes(tmp_path):
    tiled_image = make_tiled_image(
        tmp_path,
        boxes=[
            (1, 60, 10, 100, 50),  # completely inside the tile -> keeps its class
            (0, 40, 10, 80, 50),  # cut by the left border of the tile -> border
            (2, 140, 10, 160, 50),  # cut by the right border of the tile -> border
            (2, 180, 10, 220, 50),  # outside of the tile -> not in this tile
        ],
    )
    region = TileRegion(y_min=0, x_min=50, y_max=100, x_max=150)
    boxes = list(tiled_image.annotations_for(region))
    assert boxes[0] == BoundingBox(1, 10, 10, 50, 50)
    assert boxes[1] == BoundingBox(3, 0, 10, 30, 50)
    assert boxes[2] == BoundingBox(3, 90, 10, 100, 50)
    assert len(boxes) == 3


def test_annotations_for_tile_border_margin(tmp_path):
    boxes = [(1, 10, 10, 30, 30)]
    tiled_image = make_tiled_image(tmp_path, boxes=boxes)
    region = TileRegion(y_min=0, x_min=0, y_max=100, x_max=100)
    # without margin the box keeps its class
    assert tiled_image.annotations_for(region)[0].label == 1
    # with a margin of 20 pixels the box is too close to the border
    tiled_image.options = TilingOptions(tile_size=100, overlap=0.0, border_margin=20)
    assert tiled_image.annotations_for(region)[0].label == 3


def test_tile_dataset(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    tifffile.imwrite(input_dir / "slide.tif", np.zeros((200, 200, 3), dtype=np.uint8))
    # a box completely inside the upper left tile -> keeps its class
    # a box crossing the border between the two upper tiles -> becomes border
    # a box completely inside the lower right tile -> keeps its class
    (input_dir / "slide.txt").write_text(
        "1 0.1 0.1 0.1 0.1\n"  # pixels x 10-30,   y 10-30
        "0 0.5 0.25 0.1 0.1\n"  # pixels x 90-110,  y 40-60
        "2 0.8 0.8 0.1 0.1\n"  # pixels x 150-170, y 150-170
    )
    mapping = tile_dataset(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        tile_size=100,
        overlap=0.0,
        border_class=3,
    )
    # 2 x 2 tiles
    assert len(mapping) == 4
    assert len(list(output_dir.glob("slide_tile_*.tif"))) == 4
    assert len(list(output_dir.glob("slide_tile_*.txt"))) == 4

    # the tiles have the expected size
    tile = tifffile.imread(output_dir / "slide_tile_000.tif")
    assert tile.shape == (100, 100, 3)

    # upper left tile: the inner box keeps its class, the cut box becomes border
    boxes = AnnotationSet.from_yolo_file(
        output_dir / "slide_tile_000.txt", image_width=100, image_height=100
    )
    assert len(boxes) == 2
    assert boxes[0] == BoundingBox(1, 10, 10, 30, 30)
    assert boxes[1] == BoundingBox(3, 90, 40, 100, 60)

    # upper right tile: only the cut box, clipped to the tile
    boxes = AnnotationSet.from_yolo_file(
        output_dir / "slide_tile_001.txt", image_width=100, image_height=100
    )
    assert len(boxes) == 1
    assert boxes[0] == BoundingBox(3, 0, 40, 10, 60)

    # lower left tile: no annotation at all
    boxes = AnnotationSet.from_yolo_file(
        output_dir / "slide_tile_002.txt", image_width=100, image_height=100
    )
    assert len(boxes) == 0

    # lower right tile: the box keeps its class
    boxes = AnnotationSet.from_yolo_file(
        output_dir / "slide_tile_003.txt", image_width=100, image_height=100
    )
    assert len(boxes) == 1
    assert boxes[0] == BoundingBox(2, 50, 50, 70, 70)


def test_tiling_dataset_class(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    tifffile.imwrite(input_dir / "slide.tif", np.zeros((200, 200, 3), dtype=np.uint8))
    (input_dir / "slide.txt").write_text("1 0.1 0.1 0.1 0.1\n")
    dataset = TilingDataset(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        tile_size=100,
        overlap=0.0,
        border_class=3,
    )
    assert [p.name for p in dataset.image_files()] == ["slide.tif"]
    assert len(dataset.run()) == 4
    assert dataset.options.border_class == 3


def test_tile_dataset_without_annotation(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    tifffile.imwrite(input_dir / "slide.tif", np.zeros((200, 200, 3), dtype=np.uint8))
    mapping = tile_dataset(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        tile_size=100,
        overlap=0.0,
    )
    assert mapping == {}
    assert not list(output_dir.glob("*.tif"))


def test_tile_dataset_image_smaller_than_tile(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    tifffile.imwrite(input_dir / "slide.tif", np.zeros((50, 60, 3), dtype=np.uint8))
    (input_dir / "slide.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    mapping = tile_dataset(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        tile_size=100,
        overlap=0.0,
    )
    assert len(mapping) == 1
    tile = tifffile.imread(output_dir / "slide_tile_000.tif")
    assert tile.shape == (50, 60, 3)
    boxes = AnnotationSet.from_yolo_file(
        output_dir / "slide_tile_000.txt", image_width=50, image_height=60
    )
    assert len(boxes) == 1
    assert boxes[0].label == 1


def test_tile_dataset_does_not_overwrite(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    tifffile.imwrite(input_dir / "slide.tif", np.zeros((200, 200, 3), dtype=np.uint8))
    (input_dir / "slide.txt").write_text("1 0.1 0.1 0.1 0.1\n")
    tile_dataset(
        input_dir=str(input_dir), output_dir=str(output_dir), tile_size=100, overlap=0.0
    )
    annotation_path = output_dir / "slide_tile_000.txt"
    annotation_path.write_text("manually corrected\n")
    tile_dataset(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        tile_size=100,
        overlap=0.0,
        overwrite=False,
    )
    # the manual correction is still there
    assert annotation_path.read_text() == "manually corrected\n"
    tile_dataset(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        tile_size=100,
        overlap=0.0,
        overwrite=True,
    )
    assert annotation_path.read_text() != "manually corrected\n"
