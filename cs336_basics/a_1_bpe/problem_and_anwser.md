# Assignment 2. BPE
## Problem(unicode1): Understanding Unicode(1 point)
### (a) What Unicode character does chr(0) return?
Deliverable: 
chr(0)实际上返回的是空字符，是一个控制字符，在打印的时候不显示。它的值是\x00
```python
repr = chr(0)
print(f"repr: {repr.__repr__()}")
print(f"chr(0): {chr(0)}")
```


### (b) How does this character’s string representation (__repr__()) differ from its printed representation?
Deliverable:
打印出来的是什么都没有（空字符）
__repr__()返回的是：\x00，表示它在utf-8的编码信息

### (c) What happens when this character occurs in text? It may be helpful to play around with the 
following in your Python interpreter and see if it matches your expectations:
```shell
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
输出是：
'this is a test\x00string'
this is a teststring
```


## Problem(unicode2): Unicode Encodings(3 points)
### (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.
Deliverable: 
utf-8是兼容ascii编码的，英文和常见符号只需要一个字节， 如果是utf-16或者utf-32，通常就需要两个或者四个字节。相比之下，用utf-8在训练英文和大量互联网文本语料的时候，token信息会更加紧凑，训练出来的tokenizer更省空间，最终训练效率会更高
### (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```
```shell
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
Deliverable: 
因为utf-8编码是不定长的，当遇到ascii字符的时候，跟ascii保持一致（用一个字节）。当遇到非ascii字符的时候，会用多个字节表示。这里遍历了bytestring，会强行把一个由x字节组成的字符拆成单字节，结果就不对了。输入"你好吗"，很可能会报错，这个跟UTF8的编码规则有关系
UTF-8编码的规则：
ASCII 字符：第一个 bit 是 0
2 字节字符：第一个字节以 110 开头
3 字节字符：第一个字节以 1110 开头
4 字节字符：第一个字节以 11110 开头
后续字节统一以 10 开头
逐个字节打印的时候，遇到不满足规则的字节就会报错

### (c) Give a two-byte sequence that does not decode to any Unicode character(s).
Deliverable: 
就像上一题的答案一样，只要不满足这个编码规范的字节就可以了(两个字节都是1开头，但是第一个字节不是110开头，就可以)
1010 1010 1010 1111



## Problem(train_bpe_tinystories)
### (a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?
Resource requirements: ≤ 30 minutes (no GPUs), ≤ 30 GB RAM
Hint: You should be able to get under 2 minutes for BPE training using multiprocessing during pre-tokenization and the following two facts:
a) The <|endoftext|> token delimits documents in the data files
b) The <|endoftext|> token is handled as a special case before the BPE merges are applied.
```bash
vocab_size: 10000
最长的token:  accomplishment
Max RSS: 271.66015625 MB
总耗时: 37.9819 s
用户态 CPU 时间: 15.971055s
内核态 CPU 时间: 1.420937s
# 之所以只需要37秒，主要是因为我的设备性能比较好（Intel(R) Xeon(R) Platinum 8457C）

```

### (b) Profile your code. What part of the tokenizer training process takes the most time? 
_find_pair和merge_pair是耗时最多的，单次调用的时间都很短，但是调用次数很高

## Problem (train_bpe_expts_owt):  BPE Training on OpenWebText (2 points)

### (a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?
Resource requirements: ≤ 12 hours (no GPUs), ≤ 100 GB RAM
Todo: 下载owt-sample数据集, 下载完成后再回来继续做

### (b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
Deliverable: A one-to-two sentence response
Todo: 下载owt-sample数据集, 下载完成后再回来继续做

## Problem (tokenizer):  Implementing the tokenizer (15 points)
Deliverable: Implement a Tokenizer class that, given a vocabulary and a list of merges, encodes text into integer IDs and decodes integer IDs into text. Your tokenizer should also support userprovided special tokens (appending them to the vocabulary if they aren’t already there). We recommend the following interface:
```python
def __init__(self, vocab, merges, special_tokens=None) 
"""
Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept the following parameters:
- vocab: dict[int, bytes]  
- merges: list[tuple[bytes, bytes]]  
- special_tokens: list[str] | None = None  
"""
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) 
"""
Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens. This method should accept the following additional parameters:
- vocab_filepath: str  
- merges_filepath: str  
- special_tokens: list[str] | None = None  
"""
def encode(self, text: str) -> list[int] 
"""
Encode an input text into a sequence of token IDs.
"""
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
"""
Given an iterable of strings (e.g., a Python file handle)
return a generator that lazily yields token IDs. This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
"""
def decode(self, ids: list[int]) -> str
"""
Decode a sequence of token IDs into text
"""
```