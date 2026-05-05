import os
from typing import BinaryIO

from collections import Counter
import regex as re
from multiprocessing import Pool
import time

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # Pre-Tokenization
    text_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    block_pattern = "|".join(re.escape(token) for token in special_tokens)

    pre_tokenize_count = Counter()  # word -> frequency
    texts = re.split(block_pattern, chunk)
    for text in texts:
        for match in re.finditer(text_pattern, text):
            pre_tokenize_count[match.group(0)] += 1

    return pre_tokenize_count


if __name__ == "__main__":
    input_path = "/Users/weixin/Desktop/assignment-basics/data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]

    ## Usage
    stime = time.perf_counter()
    tasks = []
    with open(input_path, "rb") as f:
        num_processes = 10
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # Each session open the file by itself
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((input_path, start, end, special_tokens))

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, tasks)

    total = Counter()
    for c in results:
        total.update(c)

    print(total.most_common(10))
    etime = time.perf_counter()
    print(f"Time cost of Parallel Pre-Tokenization: {etime - stime}")
        # # The following is a serial implementation, but you can parallelize this
        # # by sending each start/end pair to a set of processes.
        # for start, end in zip(boundaries[:-1], boundaries[1:]):
        #     f.seek(start)
        #     chunk = f.read(end - start).decode("utf-8", errors="ignore")
        #     # Run pre-tokenization on your chunk and store the counts for each pre-token
