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
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from cs336_basics.a_1_bpe.utils import PAT

import regex as re

type PairType = tuple[bytes, bytes]

MAX_FREQ = 1
SPECIAL_TOKEN = [
    "<pad>",
    "<bos>",
    "<eos>",
    # Todo: add more special tokens if we need
]


pre_token_reg = re.compile(PAT)
       # 独立线程的任务
def _pre_tokenize_task(text_block_list: list[str]) -> Counter[str]:
    _result: Counter[str] = Counter()
    for text_block in text_block_list:
        _result.update(re.findall(pre_token_reg, text_block))
    return _result

def _pre_tokenize(input_path: str | os.PathLike, special_tokens: list[str]) -> Counter[str]:
    # 不支持空的special_tokens
    if not special_tokens:
        raise ValueError("special_tokens must be non-empty")


    read_file_step = 5 * 1024 * 1024  # 5 MiB
    max_workers = os.cpu_count() or 1

    # 先排序，让正则优先匹配更长的序列。再对特殊字符做处理，避免在生成正则表达式的时候产生歧义
    escaped_special_tokens = [
        re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)
    ]
    special_token_reg = re.compile("|".join(escaped_special_tokens))

    result: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        with open(input_path, encoding="utf-8") as f:
            buffered_text = ""
            _pendding_tasks: list[Future[Counter[str]]] = []
            _task_count = 0
            _finished_count = 0
            chunk = ""
            while True:
                new_text = f.read(read_file_step)
                if not new_text:
                    # 最后一次读取，直接处理缓冲区中的内容
                    chunk = buffered_text
                    buffered_text = ""
                else:
                    buffered_text += new_text
                    last_special_match = None
                    for match in special_token_reg.finditer(buffered_text):
                        last_special_match = match
                    # 当没有特殊字符时，再读取一次文件，继续处理，这里有一些重复处理的内容，有空再优化，先尽快进入后面的学习
                    if last_special_match is None:
                        continue
                    split_at = last_special_match.end()
                    chunk = buffered_text[:split_at]
                    buffered_text = buffered_text[split_at:]

                # 到这里，chunk还是空的，那就说明文件中所有的内容都被读取了，可以结束循环了
                if not chunk:
                    break
                splited_block: list[str] = re.split(special_token_reg, chunk)
                _pendding_tasks.append(executor.submit(_pre_tokenize_task, splited_block))
                _task_count += 1

                if len(_pendding_tasks) >= max_workers:
                    done, pending_set = wait(_pendding_tasks, return_when=FIRST_COMPLETED)
                    _pendding_tasks = list(pending_set)
                    for future in done:
                        result.update(future.result())
                        _finished_count += 1
                        print(f"task_count:{_task_count}, finished_count:{_finished_count}")
            # 最后清一波库存
            if len(_pendding_tasks) > 0:
                done, _ = wait(_pendding_tasks, return_when=ALL_COMPLETED)
                for future in done:
                    result.update(future.result())
                    _finished_count += 1
                    print(f"task_count:{_task_count}, finished_count:{_finished_count}")
            assert _finished_count == _task_count

    return result


def _insert_vocab(idx: int, token: bytes, vocab: dict[int, bytes]):
    vocab[idx] = token

# 用堆来管理pair的频率，边消费、边排序
class _PairHeapEntry:
    __slots__ = ("frequency", "pair")

    def __init__(self, frequency: int, pair: PairType):
        self.frequency = frequency
        self.pair = pair

    def __lt__(self, other: "_PairHeapEntry") -> bool:
        # 找出频率最高的pair，频率一样的时候比较字典序
        return (self.frequency, self.pair) > (other.frequency, other.pair)

def _count_pairs(token_list: list[bytes]) -> Counter[PairType]:
    return Counter(zip(token_list, token_list[1:]))


def _merge_pair(token_list: list[bytes], pair: PairType) -> None:
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
    pair_counts: dict[PairType, int],
) -> tuple[PairType, int] | None:
    while pair_heap:
        entry = heapq.heappop(pair_heap)
        if pair_counts.get(entry.pair) == entry.frequency:
            return entry.pair, entry.frequency
    return None

# output: vocab, merges
def train_bpe_impl(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[PairType]]:
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

    pre_token_map:Counter[str] = _pre_tokenize(input_path, special_tokens)
    # pre-tokenize
    # with open(input_path) as f:
    #     texts = f.read()
    #     splited_lines = []
    #     if len(special_tokens) > 0:
    #         sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    #         sorted_special_tokens = [re.escape(item) for item in sorted_special_tokens]
    #         special_token_reg = re.compile("|".join(sorted_special_tokens))
    #         splited_lines.extend(re.split(special_token_reg, texts))
    #     else:
    #         splited_lines.append(texts)

    #     for line in splited_lines:
    #         line_reg = re.findall(PAT, line)
    #         pre_token_map.update(line_reg)

    # 这是一个二维数组，处理token，还要处理预分词的边界
    token_list = []
    for pre_token, pr_token_freq in pre_token_map.items():
        encoded_token = pre_token.encode("utf-8")
        _arr = [bytes([_byte]) for _byte in encoded_token]
        token_list.append((_arr, pr_token_freq))

    # pair 频次只全量计算一次；后续每轮仅更新包含被合并 pair 的预分词项。
    pair_counts: dict[PairType, int] = {}
    pair_to_words: dict[PairType, set[int]] = defaultdict(set)
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
