import regex as re
import os
import time
from collections import defaultdict

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
        text_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        stime_read_file = time.perf_counter()
        blocks = self._chunk_documents_streaming(path=input_path) # Read the text content
        etime_read_file = time.perf_counter()
        print(f"Time cost of reading documents: {etime_read_file - stime_read_file}")

        pre_tokenize_count = defaultdict()         # word -> frequency
        pre_tokenize_encodings = defaultdict()     # word -> UTF-8 encoding
        stime_pre_tokenize = time.perf_counter()
        for i, block in enumerate(blocks):
            # Split the block via the "special_token"
            block_pattern = "|".join(re.escape(token) for token in special_tokens)
            texts = re.split(block_pattern, block)
            for text in texts:
                for match in re.finditer(text_pattern, text):
                    if match.group(0) not in pre_tokenize_count.keys():
                        pre_tokenize_count[match.group(0)] = 1
                    else:
                        pre_tokenize_count[match.group(0)] += 1
        etime_pre_tokenize = time.perf_counter()
        print(f"Time cost of Pre-Tokenization: {etime_pre_tokenize - stime_pre_tokenize}")

        for word in pre_tokenize_count.keys():
            pre_tokenize_encodings[word] = list(word.encode("utf-8"))

        # 3. Compute BPE merges
        merges = []
        time_byte_pair_count = 0
        time_byte_pair_max = 0
        time_update_vocab = 0
        while size < vocab_size:
            # Construct byte pairs and count
            stime = time.perf_counter()
            byte_pairs_counts = defaultdict()
            for word, count in pre_tokenize_count.items():
                encoding = pre_tokenize_encodings[word]
                for index in range(len(encoding)-1):
                    byte_pair = (encoding[index], encoding[index + 1])
                    if byte_pair not in byte_pairs_counts:
                        byte_pairs_counts[byte_pair] = count
                    else:
                        byte_pairs_counts[byte_pair] += count
            etime = time.perf_counter()
            time_byte_pair_count += etime - stime

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

            # Update the encodings of the pre-tokenized vocab
            stime = time.perf_counter()
            for word, encoding in pre_tokenize_encodings.items():
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
            etime = time.perf_counter()
            time_update_vocab += etime - stime

        print(f"Time cost of Byte-Pair Counting: {time_byte_pair_count}")
        print(f"Time cost of Finding the Byte-Pair with Highest Frequencies: {time_byte_pair_max}")
        print(f"Time cost of Updating Vocab: {time_update_vocab}")
        return vocab, merges

    def run_train_bpe_increment(self,
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
        text_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        stime_read_file = time.perf_counter()
        blocks = self._chunk_documents_streaming(path=input_path) # Read the text content
        etime_read_file = time.perf_counter()
        print(f"Time cost of reading documents: {etime_read_file - stime_read_file}")

        pre_tokenize_count = defaultdict()         # word -> frequency
        pre_tokenize_encodings = defaultdict()     # word -> UTF-8 encoding
        stime_pre_tokenize = time.perf_counter()
        for i, block in enumerate(blocks):
            # Split the block via the "special_token"
            block_pattern = "|".join(re.escape(token) for token in special_tokens)
            texts = re.split(block_pattern, block)
            for text in texts:
                for match in re.finditer(text_pattern, text):
                    if match.group(0) not in pre_tokenize_count.keys():
                        pre_tokenize_count[match.group(0)] = 1
                    else:
                        pre_tokenize_count[match.group(0)] += 1
        etime_pre_tokenize = time.perf_counter()
        print(f"Time cost of Pre-Tokenization: {etime_pre_tokenize - stime_pre_tokenize}")

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
    def _chunk_documents_streaming(
            path: str,
            chunk_size: int = CHUNK_SIZE,
            special_token: str = "<|endoftext|>"
    ):
        """
        Reads 'path' in streaming fashion, yielding chunks of text that
        each end on a '<|endoftext|>' boundary.
        """

        leftover = ""
        token_len = len(special_token)

        with open(path, "r", encoding="utf-8") as f:
            while True:
                # read one chunk_size block of text
                block = f.read(chunk_size)
                if not block:
                    # no more data in file
                    break

                # combine leftover from previous iteration + new block
                block = leftover + block
                leftover = ""

                # find the *last* occurrence of the special token in 'block'
                last_eot_idx = block.rfind(special_token)

                if last_eot_idx == -1:
                    # no complete document in this chunk
                    # keep everything in leftover for the next read
                    leftover = block
                else:
                    # up through last_eot_idx is a complete set of docs
                    yield block[: last_eot_idx + token_len]
                    # keep everything after that boundary as leftover
                    leftover = block[last_eot_idx + token_len:]

        # yield leftover text
        if leftover:
            yield leftover

if __name__ == "__main__":
    trainer = BPE_Trainer()
    vocab, merges = trainer.run_train_bpe(
        input_path="/Users/weixin/Desktop/assignment-basics/data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )
    pass