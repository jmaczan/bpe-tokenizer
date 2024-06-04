# bpe-tokenizer

Byte-Pair Encoding tokenizer for large language models that can be trained on arbitrarily huge datasets

<figure>
<div align="center">
<a href="https://anitamaczan.pl/#materia" target="_blank">
<img src="https://anitamaczan.pl/materia.jpg" width="250" alt="'Materia' by Anita Maczan, Acrylic on canvas, 40x50, 2024">
</a>
</div>
<figcaption><div align="center" style="font-family: monospace; font-size: 0.75rem">"Materia" by Anita Maczan, Acrylic on canvas, 40x50, 2024</div></figcaption>
</p>
</figure>

This implementation is suitable for working with huge datasets, because it processes data in chunks, both during tokenization and training

## Training

```py
from bpe_tokenizer import BPETokenizer

BPETokenizer().train(dataset_path="./path/to/dataset.txt")
```

### CLI

```sh
python bpe_tokenizer.py train --training_dataset path_to_your_dataset.txt --vocabulary_size 5000 --training_output path_to_output_tokenizer.json
```

defaults:

- training_dataset = "training.txt"
- vocabulary_size = 50257
- training_output = "tokenizer.json"

## Tokenize

```py
from bpe_tokenizer import BPETokenizer

BPETokenizer().tokenize(text_to_be_tokenized)
```

### CLI

```sh
python bpe_tokenizer.py tokenize --tokenizer_data path_to_tokenizer_data.json --run_data tokenize.json
```

defaults:

- tokenizer_data = "tokenizer.json"
- run_data = "tokenize.txt"

run_data file structure:

```json
{
  "data": "Study hard what interests you the most in the most undisciplined, irreverent and original manner possible - Richard Feynmann"
}
```

## Detokenize

```py
from bpe_tokenizer import BPETokenizer

BPETokenizer().detokenize(array_of_tokens_to_be_parsed_to_text)
```

### CLI

```sh
python bpe_tokenizer.py detokenize --tokenizer_data path_to_tokenizer_data.json --run_data detokenize.json
```

defaults:

- tokenizer_data = "tokenizer.json"
- run_data = "detokenize.txt"

run_data file structure:

```json
{
  "data": [
    275, 265, 282, 32, 111, 108, 100, 32, 98, 97, 115, 116, 97, 114, 100, 284,
    268, 265, 282, 32, 282, 110, 292, 98, 97, 115, 116, 97, 114, 100, 33
  ]
}
```

## License

GPL v3

Jędrzej Maczan, 2024
