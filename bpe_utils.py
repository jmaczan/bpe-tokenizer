from collections import defaultdict
from io import TextIOWrapper


def dict_to_defaultdict(d, default_type=int):
    dd = defaultdict(default_type)
    for key, value in d.items():
        dd[key] = value
    return dd


def duplicate_file(
    source_path: str, destination_path: str, chunk_size: int = 1024 * 1024
):
    """
    Duplicates a potentially large file by reading and writing in chunks.

    :param source_path: Path to the source file
    :param destination_path: Path to the destination file
    :param chunk_size: Size of each chunk to read and write (default: 1MB)
    """
    with open(source_path, "rb") as source_file:
        with open(destination_path, "wb") as dest_file:
            while True:
                chunk = source_file.read(chunk_size)
                if not chunk:
                    break
                dest_file.write(chunk)


def read_utf_8_chunk(file: TextIOWrapper, chunk_size: int = 512):
    while True:
        chunk = file.read(chunk_size)
        if not chunk:
            break
        yield chunk


def read_binary_chunk(file, token_size=4, chunk_size=512):
    bytes_chunk_size = token_size * chunk_size
    while True:
        chunk = file.read(bytes_chunk_size)
        if not chunk:
            break
        yield chunk
