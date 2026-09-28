# BPE Tokenizer Training(15 points)
# Write a function that, given a path to an input text file, trains a (byte-level) BPE
# tokenizer. Your BPE training function should handle (at least) the following input parameters:
# Input
# - input_path: str  Path to a text file with BPE tokenizer training data.
# - vocab_size: int  A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
# - special_tokens: list[str]  A list of strings to add to the vocabulary. During training,treat them as hard boundaries that prevent merges across their spans, but do not include them when computing merge statistics.

# Your BPE training function should return the resulting vocabulary and merges:
# Output
# - vocab: dict[int, bytes]  The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
# - merges: list[tuple[bytes, bytes]]  A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.

# 讲义上给出的预分词正则表达式，大致逻辑如下：
import heapq
import os
from collections import Counter, defaultdict

import regex as re

MAX_FREQ = 1
SPECIAL_TOKEN = [
    "<pad>",
    "<bos>",
    "<eos>",
    # Todo: add more special tokens if we need
]

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _insert_vocab(idx: int, token: bytes, vocab: dict[int, bytes]):
    vocab[idx] = token


class _PairHeapEntry:
    __slots__ = ("frequency", "pair")

    def __init__(self, frequency: int, pair: tuple[bytes, bytes]):
        self.frequency = frequency
        self.pair = pair

    def __lt__(self, other: "_PairHeapEntry") -> bool:
        # 找出频率最高的pair，频率一样的时候比较字典序
        return (self.frequency, self.pair) > (other.frequency, other.pair)


def _count_pairs(token_list: list[bytes]) -> Counter[tuple[bytes, bytes]]:
    return Counter(zip(token_list, token_list[1:]))


def _merge_pair(token_list: list[bytes], pair: tuple[bytes, bytes]) -> None:
    pair_0, pair_1 = pair
    merged = pair_0 + pair_1
    read_point = 0
    write_point = 0
    token_count = len(token_list)

    while read_point < token_count:
        if (
            read_point + 1 < token_count
            and token_list[read_point] == pair_0
            and token_list[read_point + 1] == pair_1
        ):
            token_list[write_point] = merged
            read_point += 2
        else:
            token_list[write_point] = token_list[read_point]
            read_point += 1
        write_point += 1

    del token_list[write_point:]


def _pop_max_pair(
    pair_heap: list[_PairHeapEntry],
    pair_counts: dict[tuple[bytes, bytes], int],
) -> tuple[tuple[bytes, bytes], int] | None:
    while pair_heap:
        entry = heapq.heappop(pair_heap)
        if pair_counts.get(entry.pair) == entry.frequency:
            return entry.pair, entry.frequency
    return None


# output: vocab, merges
def train_bpe_impl(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    global MAX_FREQ
    vocab = {}
    merges = []
    vocab_idx = 0
    # init vocab
    ## Special Token
    for special_token in special_tokens:
        _insert_vocab(vocab_idx, special_token.encode("utf-8"), vocab)
        vocab_idx += 1

    ## ASCII Character
    for byte_idx in range(256):  # 先将最小粒度的组合都放进来
        _insert_vocab(vocab_idx, bytes([byte_idx]), vocab)
        vocab_idx += 1

    pre_token_map = {}
    # pre-tokenize
    with open(input_path) as f:
        texts = f.read()
        splited_lines = []

        if len(special_tokens) > 0:
            sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
            sorted_special_tokens = [re.escape(item) for item in sorted_special_tokens]
            special_token_reg = re.compile("|".join(sorted_special_tokens))
            splited_lines.extend(re.split(special_token_reg, texts))
        else:
            splited_lines.append(texts)

        for line in splited_lines:
            line_reg = re.findall(PAT, line)
            for reg_item in line_reg:
                _count = pre_token_map.get(reg_item, 0) + 1
                pre_token_map[reg_item] = _count

    # 这是一个二维数组，处理token，还要处理预分词的边界
    token_list = []
    for pre_token, pr_token_freq in pre_token_map.items():
        encoded_token = pre_token.encode("utf-8")
        _arr = [bytes([_byte]) for _byte in encoded_token]
        token_list.append((_arr, pr_token_freq))

    # pair 频次只全量计算一次；后续每轮仅更新包含被合并 pair 的预分词项。
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for word_idx, (sub_token_list, word_freq) in enumerate(token_list):
        for pair, occurrences in _count_pairs(sub_token_list).items():
            pair_counts[pair] = pair_counts.get(pair, 0) + occurrences * word_freq
            pair_to_words[pair].add(word_idx)

    pair_heap = [_PairHeapEntry(freq, pair) for pair, freq in pair_counts.items()]
    heapq.heapify(pair_heap)

    while len(vocab) < vocab_size:
        max_item = _pop_max_pair(pair_heap, pair_counts)
        if max_item is None:
            break
        max_pair, freq = max_item
        if freq < MAX_FREQ:
            break

        merges.append(max_pair)
        _insert_vocab(vocab_idx, max_pair[0] + max_pair[1], vocab)
        vocab_idx += 1

        affected_words = tuple(pair_to_words.pop(max_pair, ()))
        changed_pairs = {max_pair}

        for word_idx in affected_words:
            sub_token_list, word_freq = token_list[word_idx]
            old_counts = _count_pairs(sub_token_list)
            _merge_pair(sub_token_list, max_pair)
            new_counts = _count_pairs(sub_token_list)
            changed_pairs.update(old_counts)
            changed_pairs.update(new_counts)

            for pair, occurrences in old_counts.items():
                pair_counts[pair] -= occurrences * word_freq
                word_ids = pair_to_words.get(pair)
                if word_ids is not None:
                    word_ids.discard(word_idx)
                    if not word_ids:
                        pair_to_words.pop(pair, None)

            for pair, occurrences in new_counts.items():
                pair_counts[pair] = pair_counts.get(pair, 0) + occurrences * word_freq
                pair_to_words[pair].add(word_idx)

        for pair in changed_pairs:
            pair_freq = pair_counts.get(pair, 0)
            if pair_freq > 0:
                heapq.heappush(pair_heap, _PairHeapEntry(pair_freq, pair))
            else:
                pair_counts.pop(pair, None)

    return vocab, merges
