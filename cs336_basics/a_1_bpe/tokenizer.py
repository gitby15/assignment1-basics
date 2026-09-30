import json
import time
import resource
from collections.abc import Iterable, Iterator
import regex as re

from cs336_basics.a_1_bpe.train_bpe import train_bpe_impl
from pathlib import Path
from cs336_basics.a_1_bpe.utils import PAT, SPECIAL_TOKENS





def train_tokenizer(data_file: Path, tokenizer_folder: Path):
    # 获取当前进程及其子进程的资源使用统计
    start_ts = time.time()
    vocab, merges = train_bpe_impl(data_file, 10000, SPECIAL_TOKENS)
    end_ts = time.time()
    usage = resource.getrusage(resource.RUSAGE_SELF)

    longest_token = max(vocab.values(), key=lambda x: len(x)).decode("utf-8")

    print(f"vocab_size: {len(vocab)}")
    print(f"最长的token: {longest_token}")
    # 注意：Linux 上单位是 KB，macOS 上单位是 Byte，我的是Linux
    max_rss = usage.ru_maxrss
    print(f"Max RSS: {max_rss/1024} MB")
    print(f"总耗时: {end_ts - start_ts:.4f} s")
    print(f"用户态 CPU 时间: {usage.ru_utime}s")
    print(f"内核态 CPU 时间: {usage.ru_stime}s")
    vocab_path = tokenizer_folder / "vocab.json"
    merges_path = tokenizer_folder / "merges.json"
    special_tokens_path = tokenizer_folder / "special_tokens.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(
            {str(token_id): token.hex() for token_id, token in vocab.items()},
            f,
        )
    with merges_path.open("w", encoding="utf-8") as f:
        json.dump(
            [[left.hex(), right.hex()] for left, right in merges],
            f,
        )
    with special_tokens_path.open("w", encoding="utf-8") as f:
        json.dump(
            SPECIAL_TOKENS,
            f,
        )
    print(f"tokenizer 已写入: {tokenizer_folder}")


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []

        vocab_values = set(self.vocab.values())
        for special_token in self.special_tokens:
            encoded_token = special_token.encode("utf-8")
            if encoded_token not in vocab_values:
                self.vocab[len(self.vocab)] = encoded_token
                vocab_values.add(encoded_token)

        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        # 记录每一个pair和他的顺序
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        escaped_special_tokens = [
            re.escape(token)
            for token in sorted(self.special_tokens, key=len, reverse=True)
        ]
        self.special_tokens_reg = (
            re.compile(f"({'|'.join(escaped_special_tokens)})")
            if escaped_special_tokens
            else None
        )

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with Path(vocab_filepath).open("r", encoding="utf-8") as f:
            serialized_vocab = json.load(f)
        with Path(merges_filepath).open("r", encoding="utf-8") as f:
            serialized_merges = json.load(f)

        vocab = {
            int(token_id): bytes.fromhex(token)
            for token_id, token in serialized_vocab.items()
        }
        merges = [
            (bytes.fromhex(left), bytes.fromhex(right))
            for left, right in serialized_merges
        ]
        return cls(vocab, merges, special_tokens)




    def encode(self, text: str) -> list[int]:
        chunks = (
            self.special_tokens_reg.split(text)
            if self.special_tokens_reg is not None
            else [text]
        )
        special_tokens = set(self.special_tokens)
        result: list[int] = []

        for chunk in chunks:
            if not chunk:
                continue
            # special token不拼接，直接映射
            if chunk in special_tokens:
                result.append(self.token_to_id[chunk.encode("utf-8")])
                continue

            
            for match in re.finditer(PAT, chunk):
                encoded = match.group().encode("utf-8")
                # 把pre_token拆成多个字节
                tokens = [bytes([byte]) for byte in encoded]

                # 对单个token完成一次merge，每个token都会尝试拼装到最长的结果
                # 对于merges数组，需要从左往右依次拼接
                while len(tokens) >= 2:

                    # 找出优先级最高的pair，如果没找到，就会返回前两个token
                    best_pair = min(
                        zip(tokens, tokens[1:]),
                        key=lambda pair: self.merge_ranks.get(pair, float("inf")),
                    )
                    # 如果前两个token也不是pair，就不用继续往下找了
                    if best_pair not in self.merge_ranks:
                        break

                    merged_tokens: list[bytes] = []
                    index = 0
                    tokens_len = len(tokens)
                    while index < tokens_len:
                        if (
                            index + 1 < tokens_len
                            and (tokens[index], tokens[index + 1]) == best_pair
                        ):
                            merged_tokens.append(best_pair[0] + best_pair[1])
                            index += 2
                        else:
                            merged_tokens.append(tokens[index])
                            index += 1
                    tokens = merged_tokens

                result.extend(self.token_to_id[token] for token in tokens)
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")




if __name__ == "__main__":
    # tinystory_path = Path.cwd() / "data" / "TinyStoriesV2-GPT4-train.txt"
    tokenizer_folder = Path(__file__).parent / "tinystories"
    # train_tokenizer(tinystory_path, tokenizer_folder)
    tokenizer = Tokenizer.from_files(tokenizer_folder / "vocab.json", tokenizer_folder / "merges.json", SPECIAL_TOKENS)
    tokenizer.encode("Hello, I am a student.")
