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
import os

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
    if vocab.get(idx, None) is not None:
        raise ValueError(f"idx {idx} already exists in vocab")
    vocab[idx] = token


def _find_pairs(token_list: list[bytes], pairs: dict[tuple[bytes, bytes], int], count: int) -> dict[tuple[bytes, bytes], int]:

    _len = len(token_list) - 1
    for idx in range(_len):
        pair = (token_list[idx], token_list[idx + 1])
        pairs[pair] = pairs.get(pair, 0) + count

    return pairs


def _find_max_pair(pairs: dict[tuple[bytes, bytes], int]) -> tuple[tuple[bytes, bytes], int]:
    # 找到频率最高的pair
    # 如果有多个频率一样高的pair，选字典序最大的
    max_pair, max_freq = max(
        pairs.items(),
        key=lambda item: (
            item[1],
            item[0],
        ),
    )
    return max_pair, max_freq


def _merge_pair(token_list, pair: tuple[bytes, bytes]):
    pair_0 = pair[0]
    pair_1 = pair[1]
    for (sub_token_list, _freq) in token_list:
        idx = 0
        while True:
            if idx >= len(sub_token_list) - 1:
                break
            current = sub_token_list[idx]
            next = sub_token_list[idx + 1]
            if current == pair_0 and next == pair_1:
                # 合并pair_0和pair_1
                sub_token_list[idx] = pair_0 + pair_1
                del sub_token_list[idx + 1]
                continue
            idx += 1


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
    for (pre_token, pr_token_freq) in pre_token_map.items():
        encoded_token = pre_token.encode("utf-8")
        _arr = [bytes([_byte]) for _byte in encoded_token]   
        token_list.append((_arr, pr_token_freq))

    # 中止条件：1)vocab满了，2)频率小于xx，3)没有pair了

    while True:
        token_pairs = {}
        # 每个元素是一个tuple：(pair, count)
        pair_maps = []

        for (sub_token_list, count) in token_list:
            token_pairs = _find_pairs(sub_token_list, token_pairs, count)

        for (pair, count) in pair_maps:
            token_pairs[pair] = token_pairs.get(pair, 0) + count


        max_pair, freq = _find_max_pair(token_pairs)
        merges.append(max_pair)
        _insert_vocab(vocab_idx, max_pair[0] + max_pair[1], vocab)
        vocab_idx += 1
        _merge_pair(token_list, max_pair)

        if len(vocab) >= vocab_size:
            break
        if freq <= MAX_FREQ:
            break
        # 如果下一轮找不到pair了，就结束
        if len(token_pairs) < 2:
            print("Token Pairs Break")
            break

    return vocab, merges


# 目前的性能是1.19秒完成测试
# 其实还有挺多可以优化的空间的，比如说对于token_list有大量可以跳过的扫描、多线程等
# 但是为了更快进入后面的学习步骤，我就先不继续优化了