import regex as re
import os
import time
from collections import defaultdict, Counter
from typing import BinaryIO
from multiprocessing import Pool

INIT_SIZE = 256
CHUNK_SIZE = 1024 * 1024 * 8

class BPE_Trainer():
    def run_train_bpe(self,
            input_path: str | os.PathLike,
            vocab_size: int,
            special_tokens: list[str],
            **kwargs,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Given the path to an input corpus, run train a BPE tokenizer and
        output its vocabulary and merges.

        Args:
            input_path (str | os.PathLike): Path to BPE tokenizer training data.
            vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
            special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be
                kept as a single token. If these special tokens occur in the `input_path`,
                they are treated as any other string.

        Returns:
            tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
                vocab:
                    The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                    to bytes (token bytes)
                merges:
                    BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                    representing that <token1> was merged with <token2>.
                    Merges are ordered by order of creation.
        """

        # 1. Vocab Initialization
        vocab = defaultdict()
        for index in range(INIT_SIZE):
            vocab[index] = bytes([index])
        for index, special_token in enumerate(special_tokens):
            vocab[index + INIT_SIZE] = special_token.encode('utf-8')
        size = len(vocab)

        # 2. Pre-Tokenization
        stime = time.perf_counter()
        tasks = []
        with open(input_path, "rb") as f:
            num_processes = 10
            boundaries = self.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            # Each session open the file by itself
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                tasks.append((input_path, start, end, special_tokens))

        with Pool(processes=num_processes) as pool:
            results = pool.map(self.process_chunk, tasks)

        pre_tokenize_count = Counter()
        pre_tokenize_encodings = Counter()
        for result in results:
            pre_tokenize_count.update(result)

        etime = time.perf_counter()
        print(f"Time cost of Parallel Pre-Tokenization: {etime - stime}")

        for word in pre_tokenize_count.keys():
            pre_tokenize_encodings[word] = list(word.encode("utf-8"))

        # 3. Increment Update
        byte_pair_to_words = defaultdict(set)
        byte_pairs_counts = defaultdict(int)
        for word, count in pre_tokenize_count.items():
            encoding = pre_tokenize_encodings[word]
            for index in range(len(encoding) - 1):
                byte_pair = (encoding[index], encoding[index + 1])

                # Pair Counting
                if byte_pair not in byte_pairs_counts:
                    byte_pairs_counts[byte_pair] = count
                else:
                    byte_pairs_counts[byte_pair] += count

                # Construct Pair-to-Word index
                if byte_pair not in byte_pair_to_words.keys():
                    byte_pair_to_words[byte_pair] = set()
                byte_pair_to_words[byte_pair].add(word)

        # 4. Compute BPE merges
        merges = []
        time_byte_pair_increment_update = 0
        time_byte_pair_max = 0
        while size < vocab_size:
            if len(byte_pairs_counts) == 0:
                break

            # Select the byte pair with the highest frequency
            stime = time.perf_counter()
            byte_pair_max, max_count = max(byte_pairs_counts.items(), key=lambda x: (x[1], (vocab[x[0][0]], vocab[x[0][1]])))
            etime = time.perf_counter()
            time_byte_pair_max += etime - stime

            # Merge into the initial vocab
            merged_bytes = vocab[byte_pair_max[0]] + vocab[byte_pair_max[1]]
            vocab[size] = merged_bytes
            merges.append((vocab[byte_pair_max[0]], vocab[byte_pair_max[1]]))
            new_token = size
            size += 1

            stime = time.perf_counter()
            # Update Byte-Pair Counting & encoding of the pre-tokenized texts
            affected_words = list(byte_pair_to_words[byte_pair_max])
            for word in affected_words:
                encoding = pre_tokenize_encodings[word]
                count = pre_tokenize_count[word]

                # Update Old Byte-Pair Counting & Pair-to-Words index
                for index in range(len(encoding) - 1):
                    byte_pair = (encoding[index], encoding[index + 1])
                    if byte_pair in byte_pairs_counts:
                        byte_pairs_counts[byte_pair] -= count
                        if byte_pairs_counts[byte_pair] <= 0:
                            del byte_pairs_counts[byte_pair]

                    # Remove Old Pair-to-Words index
                    if byte_pair in byte_pair_to_words.keys():
                        byte_pair_to_words[byte_pair].discard(word)
                        if len(byte_pair_to_words[byte_pair]) == 0:
                            byte_pair_to_words.pop(byte_pair)

                # Update Encodings
                new_encoding = []
                index = 0
                has_new_token = False
                while index < len(encoding):
                    if index < len(encoding) - 1 and ((encoding[index], encoding[index + 1]) == byte_pair_max):
                        new_encoding.append(new_token)
                        index += 2
                        has_new_token = True
                    else:
                        new_encoding.append(encoding[index])
                        index += 1
                if has_new_token:
                    pre_tokenize_encodings[word] = new_encoding

                # Find New Byte Pair
                for index in range(len(new_encoding) - 1):
                    byte_pair = (new_encoding[index], new_encoding[index + 1])

                    # Pair Counting
                    if byte_pair not in byte_pairs_counts:
                        byte_pairs_counts[byte_pair] = pre_tokenize_count[word]
                    else:
                        byte_pairs_counts[byte_pair] += pre_tokenize_count[word]

                    # Construct Pair-to-Word index
                    if byte_pair not in byte_pair_to_words.keys():
                        byte_pair_to_words[byte_pair] = set()
                    byte_pair_to_words[byte_pair].add(word)

            etime = time.perf_counter()
            time_byte_pair_increment_update += etime - stime

        print(f"Time cost of Incrementally Byte-Pair Update: {time_byte_pair_increment_update}")
        print(f"Time cost of Finding the Byte-Pair with Highest Frequencies: {time_byte_pair_max}")
        print("num pairs:", len(byte_pairs_counts))
        print("zero pairs:", sum(1 for v in byte_pairs_counts.values() if v <= 0))
        return vocab, merges

    @staticmethod
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

    @staticmethod
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
    trainer = BPE_Trainer()
    vocab, merges = trainer.run_train_bpe(
        input_path="/Users/weixin/Desktop/assignment-basics/data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )
    pass