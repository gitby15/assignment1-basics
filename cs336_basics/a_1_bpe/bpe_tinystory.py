import time
import resource

from cs336_basics.a_1_bpe.train_bpe import train_bpe_impl
from tests.common import FIXTURES_PATH


# tinystory_path = FIXTURES_PATH / "tinystories_sample.txt"
tinystory_path = FIXTURES_PATH / "tinystories_sample_5M.txt"

def bpe_tinystory():
    # 获取当前进程及其子进程的资源使用统计
    

    start_ts = time.time()
    vocab, merges = train_bpe_impl(tinystory_path, 10000, ["<|endoftext|>"])
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


if __name__ == "__main__":
    bpe_tinystory()
