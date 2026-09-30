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

        # 保险起见，检查一次special_tokens
        for special_token in self.special_tokens:
            encoded = special_token.encode('utf-8')
            if encoded not in self.vocab.values():
                self.vocab[len(self.vocab)] = encoded

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
        self.max_token_len = max(len(token) for token in self.vocab.values()) or 0

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
        return list(self.encode_iterable([text]))

    def _handle_pre_token(self, pre_token: str) -> list[int]:
        tokens = [bytes([byte]) for byte in pre_token.encode('utf-8')]
        while len(tokens) >= 2:
            pair_list = zip(tokens, tokens[1:])
            best_pair = min(pair_list, key=lambda pair: self.merge_ranks.get(pair, float('inf')))
            if best_pair not in self.merge_ranks:
                break
            index = 0
            boundary  = len(tokens) -1
            merged_tokens = []
            while index < boundary:
                pair = (tokens[index], tokens[index + 1])
                if pair == best_pair:
                    _token = pair[0] + pair[1]
                    merged_tokens.append(_token)
                    index += 2
                else:
                    _token = tokens[index]
                    merged_tokens.append(_token)
                    index += 1
            # 补上最后一个元素
            if index == boundary:
                merged_tokens.append(tokens[-1])
            tokens = merged_tokens
        return [self.token_to_id[token] for token in tokens]

    # (token_id, splited_idx)
    def _handle_buffer_chunk(self, safety_chunk: str, skip_last = True) -> Iterator[tuple[int, int]]:
        # 这是相对safety_chunk的位置
        splited_idx = 0
        for pre_token_match in re.finditer(PAT, safety_chunk):
            chunk_len = len(safety_chunk)
            # 给尾巴保留安全距离
            if skip_last and chunk_len - pre_token_match.end() <= self.max_token_len:
                break
            pre_token = pre_token_match.group()
            splited_idx = pre_token_match.end()
            for token_id in self._handle_pre_token(pre_token):
                yield (token_id, splited_idx)
            
        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer_chunk = ""
        for text in iterable:
            buffer_chunk += text
            
            if self.special_tokens_reg is None:
                # 这是相对buffer_chunk的位置
                last_realative_idx = 0
                for token_id, _idx in self._handle_buffer_chunk(buffer_chunk):
                    last_realative_idx = _idx
                    yield token_id
                buffer_chunk = buffer_chunk[last_realative_idx:]
            
            else:
                # 这是相对buffer_chunk的位置
                last_realative_idx = 0
                for safety_chunk_match in re.finditer(self.special_tokens_reg, buffer_chunk):
                    safety_chunk = buffer_chunk[last_realative_idx:safety_chunk_match.start()]
                    special_token = buffer_chunk[safety_chunk_match.start():safety_chunk_match.end()]
                    last_realative_idx = safety_chunk_match.end()
                    for token_id, _ in self._handle_buffer_chunk(safety_chunk, False):
                        yield token_id
                    yield self.token_to_id[special_token.encode("utf-8")]
                # 如果这一次没有special_token，就先消化一批
                if last_realative_idx == 0:
                    for token_id, _idx in self._handle_buffer_chunk(buffer_chunk):
                        last_realative_idx = _idx
                        yield token_id

                    
                buffer_chunk = buffer_chunk[last_realative_idx:]
        # 收尾
        if buffer_chunk:
            for token_id, _ in self._handle_buffer_chunk(buffer_chunk, False):
                yield token_id
        


    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")




if __name__ == "__main__":
    # tinystory_path = Path.cwd() / "data" / "TinyStoriesV2-GPT4-train.txt"
    tokenizer_folder = Path(__file__).parent / "tinystories"
    # train_tokenizer(tinystory_path, tokenizer_folder)
    tokenizer = Tokenizer.from_files(tokenizer_folder / "vocab.json", tokenizer_folder / "merges.json", SPECIAL_TOKENS)
    tokenizer.encode("Hello, I am a student.")
