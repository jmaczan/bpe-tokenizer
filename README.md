# bpe-tokenizer

Byte-Pair Encoding tokenizer

This implementation is suitable for working with huge datasets, because it processes data in chunks, both during tokenization and training

```py
BPETokenizer().train(dataset_path="./path/to/dataset.txt")
```

Default values like vocabulary size are aligned with GPT-2

Jędrzej Maczan, 2024
