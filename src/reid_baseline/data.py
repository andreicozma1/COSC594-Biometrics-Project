"""Read the standard Market-1501 query and gallery folders.

Split handling:
    - Keep records in filename order so embedding rows are repeatable.
    - Drop junk identity -1 and retain gallery distractor identity 0.
    - Require known identities in the query split.

Loading records reads filenames only; image decoding happens during extraction.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# Standard release counts include junk gallery images, but not the optional 500K extension.
STANDARD_QUERY_COUNT = 3368
STANDARD_GALLERY_COUNT = 19732

# Some downloaded Market-1501 files have a repeated .jpg suffix.
_FILENAME = re.compile(r"(?P<pid>-1|\d{4})_c(?P<camera>[1-6])s\d+_\d+_\d+\.jpg(?:\.jpg)?")


@dataclass(frozen=True)
class ImageRecord:
    """Identify one image within a dataset.

    Attributes:
        path: Image path relative to the dataset root, including its split folder.
        pid: Original person identity from the filename; labels are not renumbered.
        camera: Original one-based camera ID.
    """

    path: str
    pid: int
    camera: int

    @property
    def filename(self) -> str:
        """Get the filename without its split folder.

        Returns:
            The original basename, including its extension, for display or lookup.
        """
        return Path(self.path).name


@dataclass(frozen=True)
class Market1501:
    """Hold the usable image records and original split sizes.

    Attributes:
        root: Resolved dataset directory.
        queries: Query records sorted by filename.
        gallery: Gallery records sorted by filename, including identity 0.
        raw_query_count: Number of query JPEG paths before filtering.
        raw_gallery_count: Number of gallery JPEG paths before filtering.
    """

    root: Path
    queries: list[ImageRecord]
    gallery: list[ImageRecord]
    raw_query_count: int
    raw_gallery_count: int

    @property
    def has_standard_counts(self) -> bool:
        """Compare the raw folder counts with the standard release.

        Returns:
            True when both counts match. This checks split size only; it does not
            verify that the files are the original release images.
        """
        return (
            self.raw_query_count == STANDARD_QUERY_COUNT
            and self.raw_gallery_count == STANDARD_GALLERY_COUNT
        )


def parse_filename(filename: str) -> tuple[int, int]:
    """Read the person and camera labels encoded in a Market-1501 filename.

    Args:
        filename: Basename including .jpg. A repeated .jpg suffix is also accepted.

    Returns:
        The original person ID and one-based camera ID, in that order.

    Raises:
        ValueError: The filename does not match the expected pattern.
    """
    match = _FILENAME.fullmatch(filename)
    if match is None:
        raise ValueError(
            f"Malformed Market-1501 filename: {filename}. "
            "Expected a name such as 0001_c1s1_000001_00.jpg."
        )
    return int(match["pid"]), int(match["camera"])


def _read_split(root: Path, folder: str) -> tuple[list[ImageRecord], int]:
    """Collect the usable records from one split folder.

    Args:
        root: Dataset directory used as the base for saved image paths.
        folder: query or bounding_box_test, relative to root.

    Returns:
        Records sorted by filename and the JPEG count before junk filtering.

    Raises:
        FileNotFoundError: The split folder is missing.
        ValueError: A filename is malformed, a query has identity 0, or no usable
            records remain.
    """
    directory = root / folder
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing dataset folder: {directory}. "
            "Point --dataset-root to the extracted Market-1501 directory."
        )

    # File-system order can vary; saved metadata and vectors need a repeatable order.
    paths = sorted(directory.glob("*.jpg"), key=lambda path: path.name)
    records: list[ImageRecord] = []
    for path in paths:
        pid, camera = parse_filename(path.name)
        # Junk is excluded, but identity 0 remains a gallery distractor.
        if pid == -1:
            continue
        if folder == "query" and pid == 0:
            raise ValueError(f"Query must have a known person identity: {path}")

        # Relative paths let a caller resolve all images from one dataset root.
        records.append(ImageRecord(path.relative_to(root).as_posix(), pid, camera))

    if not records:
        raise ValueError(f"No usable Market-1501 JPEGs in {directory}")

    # Keep the unfiltered count for comparison with the standard release.
    return records, len(paths)


def load_market1501(root: Path) -> Market1501:
    """Load image records for the standard query and gallery splits.

    Args:
        root: Extracted dataset directory. A leading tilde is expanded.

    Returns:
        Split records and raw counts, with paths relative to the resolved root.

    Raises:
        FileNotFoundError: query/ or bounding_box_test/ is missing.
        ValueError: A split contains invalid labels or no usable records.
    """
    root = root.expanduser().resolve()
    queries, query_count = _read_split(root, "query")
    gallery, gallery_count = _read_split(root, "bounding_box_test")
    return Market1501(root, queries, gallery, query_count, gallery_count)
