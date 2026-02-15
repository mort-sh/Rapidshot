import pytest

pytest.importorskip("comtypes")

from rapidshot.capture import ScreenCapture


class DummyOutput:
    def __init__(self, surface_size, rotation_angle):
        self._surface_size = surface_size
        self._rotation_angle = rotation_angle

    @property
    def surface_size(self):
        return self._surface_size

    @property
    def rotation_angle(self):
        return self._rotation_angle


class DummyBox:
    def __init__(self):
        self.left = self.top = self.right = self.bottom = 0


def make_capture(width, height):
    capture = object.__new__(ScreenCapture)
    capture.width = width
    capture.height = height
    capture._sourceRegion = None
    capture.shot_w = 0
    capture.shot_h = 0
    return capture


@pytest.mark.parametrize(
    "rotation, surface_size, region, expected",
    [
        (0, (8, 6), (1, 2, 4, 5), (1, 2, 4, 5)),
        (90, (6, 8), (1, 2, 4, 5), (2, 2, 5, 5)),
        (180, (8, 6), (1, 2, 4, 5), (4, 1, 7, 4)),
        (270, (6, 8), (1, 2, 4, 5), (3, 1, 6, 4)),
    ],
)
def test_region_to_memory_region(rotation, surface_size, region, expected):
    capture = make_capture(8, 6)
    output = DummyOutput(surface_size, rotation)
    assert capture.region_to_memory_region(region, rotation, output) == expected


def test_region_to_memory_region_rotation_mismatch():
    capture = make_capture(10, 10)
    output = DummyOutput((10, 10), 90)
    with pytest.raises(AssertionError):
        capture.region_to_memory_region((0, 0, 5, 5), 0, output)


def test_normalize_region_validates_without_side_effects():
    capture = make_capture(12, 8)
    normalized = capture._normalize_region((1, 2, 6, 7))
    assert normalized == (1, 2, 6, 7)
    # Calling normalize should not set capture.region
    assert not hasattr(capture, "region")


@pytest.mark.parametrize(
    "region",
    [
        (-1, 0, 2, 2),
        (0, -1, 2, 2),
        (0, 0, 13, 1),
        (0, 0, 1, 9),
        (4, 4, 3, 5),
    ],
)
def test_normalize_region_invalid(region):
    capture = make_capture(12, 8)
    with pytest.raises(ValueError):
        capture._normalize_region(region)


def test_validate_region_updates_state():
    capture = make_capture(20, 10)
    capture._sourceRegion = DummyBox()
    capture._validate_region((2, 3, 10, 9))
    assert capture.region == (2, 3, 10, 9)
    assert capture.shot_w == 8
    assert capture.shot_h == 6
    assert capture._sourceRegion.left == 2
    assert capture._sourceRegion.top == 3
    assert capture._sourceRegion.right == 10
    assert capture._sourceRegion.bottom == 9
