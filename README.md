# bpe-tokenizer

Byte-Pair Encoding tokenizer for large language models that can be trained on arbitrary huge datasets

<figure>
<div align="center">
<a href="https://anitamaczan.pl/#materia" target="_blank">
<img src="https://anitamaczan.pl/materia.jpg" style="max-width: 200px; height: auto;" alt="'Materia' by Anita Maczan, Acrylic on canvas, 40x50, 2024">
</a>
</div>
<figcaption><div align="center" style="font-family: monospace; font-size: 0.75rem">"Materia" by Anita Maczan, Acrylic on canvas, 40x50, 2024</div></figcaption>
</p>
</figure>

This implementation is suitable for working with huge datasets, because it processes data in chunks, both during tokenization and training

```py
BPETokenizer().train(dataset_path="./path/to/dataset.txt")
```

Default values like vocabulary size are aligned with GPT-2

Jędrzej Maczan, 2024
