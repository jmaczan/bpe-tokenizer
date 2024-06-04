from collections import defaultdict
from enum import Enum
import json
import tempfile
from bpe_utils import (
    dict_to_defaultdict,
    duplicate_file,
    read_binary_chunk,
    read_utf_8_chunk,
)
import os
import argparse
import numpy as np

default_vocabulary_size = 50_257


class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer
    """

    def __init__(self, vocabulary: dict = None) -> None:

        self.vocabulary = dict_to_defaultdict(
            vocabulary or {token: token for token in range(256)}
        )
        self.token_frequencies = defaultdict(int)

    def train(
        self,
        dataset_path,
        vocabulary_size: int = default_vocabulary_size,
        in_place: bool = False,
    ):
        """
        Train a tokenizer
        :param in_place: if True, then will modify the dataset when training a tokenizer, so it saves a disk space by not copying a dataset during training (bool)

        Procedure:
        We do things in chunks because datasets might be huge so we don't want to load all the dataset into memory at once
        - Reset token frequencies object
        - Count a frequency of any two subsequent tokens, like in a sentence "hey, hello" we would identify (("h", "e"), 2) and all other pairs would have 1 occurence
        - Pick a pair that has the most occurences
        - Combine those tokens and put it into vocabulary
        - Replace all occurences of the most frequent pair in dataset (working copy or original, if in_place=True) with a new token index
        - Repeat from first step until we get a vocabulary of length of vocabulary_size or when all pairs have only 1 occurence

        """

        if dataset_path is None:
            raise Exception(
                "Please specify a path to a local file containing dataset, so it can be read in chunks into the tokenizer"
            )

        if in_place:
            working_copy_path = dataset_path
        else:
            working_copy_path = "working_copy.txt"
            duplicate_file(dataset_path, working_copy_path)

        # in first iteration, turn all characters into tokens and then into binary representation

        temp_file, temp_file_path = tempfile.mkstemp()

        with open(working_copy_path, "r", encoding="utf-8") as working_copy, open(
            temp_file, "wb"
        ) as temp_copy:
            for chunk in read_utf_8_chunk(working_copy):
                chunk = chunk.encode(encoding="utf-8")

                for token in chunk:
                    temp_copy.write(
                        token.to_bytes(4, byteorder="little")
                    )  # token size in bytes (4, 3, 2 or 1) can be calculated based on vocabulary_size and then reused in all places when I operate on bytes

        os.replace(temp_file_path, working_copy_path)

        while len(self.vocabulary) < vocabulary_size and (
            len(self.token_frequencies) == 0 or not self.all_pairs_are_unique()
        ):
            self.token_frequencies = defaultdict(int)  # reset token frequencies counter

            with open(working_copy_path, "rb") as working_copy:
                for chunk in read_binary_chunk(file=working_copy):
                    tokens = np.frombuffer(
                        chunk, dtype=np.uint32
                    )  # adjust here based on vocabulary_size
                    self.count_token_frequencies(tokens)

            most_frequent_pair = self.sort_by_token_frequency(self.token_frequencies)[0]

            new_token_index = len(self.vocabulary)

            self.vocabulary[new_token_index] = (
                int(most_frequent_pair[0][0]),
                int(most_frequent_pair[0][1]),
            )

            temp_file, temp_file_path = tempfile.mkstemp()

            # replace all occurences of a most frequent pair
            with open(
                working_copy_path,
                "rb",
            ) as working_copy, open(temp_file, "wb") as temp_copy:
                for chunk in read_binary_chunk(working_copy):
                    tokens = np.frombuffer(
                        chunk, dtype=np.uint32  # adjust here based on vocabulary_size
                    )
                    # this merge code is heavily inspired by how Andrej have implemented it here: https://github.com/karpathy/minbpe/blob/master/minbpe/base.py#L25 Previously, I tried slicing arrays and putting new tokens in the middle and it was too much headache of handling indices properly
                    new_tokens = []
                    index = 0

                    while index < len(tokens):
                        if (
                            index + 1 < len(tokens)
                            and most_frequent_pair[0][0] == tokens[index]
                            and most_frequent_pair[0][1] == tokens[index + 1]
                        ):
                            new_tokens.append(
                                np.uint32(new_token_index)
                            )  # to be adjusted
                            index += 2
                        else:
                            new_tokens.append(tokens[index])
                            index += 1

                    for token in new_tokens:
                        temp_copy.write(token.tobytes())

            os.replace(temp_file_path, working_copy_path)

    def all_pairs_are_unique(self):
        return self.sort_by_token_frequency(self.token_frequencies)[0][1] == 1

    def count_token_frequencies(self, data):
        for index in range(0, len(data) - 1, 2):
            self.token_frequencies[(data[index], data[index + 1])] += 1

    def sort_by_token_frequency(self, data: dict):
        return list(sorted(data.items(), key=lambda item: item[1], reverse=True))

    def tokenize(self, data):
        tokens = list(map(int, data.encode(encoding="utf-8")))
        print(tokens)
        added_keys = [key for key in self.vocabulary.keys() if int(key) > 255]
        for vocabulary_item in added_keys:
            new_tokens = []
            index = 0

            while index < len(tokens):
                if (
                    index + 1 < len(tokens)
                    and self.vocabulary[vocabulary_item][0] == tokens[index]
                    and self.vocabulary[vocabulary_item][1] == tokens[index + 1]
                ):
                    new_tokens.append(int(vocabulary_item))
                    index += 2
                else:
                    new_tokens.append(tokens[index])
                    index += 1

            tokens = new_tokens

        return tokens

    def detokenize(self, tokens):
        # we assume data to be array of tokens from vocabulary
        # Procedure:
        # - iterate over elements from vocabulary in reverse order, so starting from last and ending on index 255 (not processing it, since 0-255 are predefined elements of our vocab)
        # - search for tokens that are like current key of element in vocabulary
        # - replace all these elements with a pairs of consecutive tokens from a value of an element in vocab
        # - repeat until you get to 255
        # - merge items into string and return it
        for vocabulary_token in sorted(
            list(map(int, self.vocabulary.keys())), reverse=True
        ):
            if vocabulary_token == 255:
                break

            index = 0
            new_tokens = []
            while index < len(tokens):
                if tokens[index] == vocabulary_token:
                    new_tokens.append(self.vocabulary[str(vocabulary_token)][0])
                    new_tokens.append(self.vocabulary[str(vocabulary_token)][1])
                else:
                    new_tokens.append(tokens[index])
                index += 1

            tokens = new_tokens

        print(tokens)
        return "".join(token for token in bytes(tokens).decode("utf-8"))

    def load_vocabulary(self, vocabulary: dict = {}):
        self.vocabulary = vocabulary

    def load(self, vocabulary: dict = {}):
        self.vocabulary = vocabulary


class BPEAction(Enum):
    train = "train"
    tokenize = "tokenize"
    detokenize = "detokenize"

    def __str__(self) -> str:
        return self.value


default_training_dataset_location = "training.txt"
default_training_output_location = "tokenizer.json"
default_inference_data_location = default_training_output_location
default_tokenize_data_location = "tokenize.txt"
default_detokenize_data_location = "detokenize.txt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Byte-Pair Encoding Tokenizer. Default training dataset location: 'training.txt'. Default tokenize data location 'tokenize.txt'. File structure: {'data': [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33, 256]}. Default detokenize data location 'detokenize.txt'. File structure: {'data': 'Hello world, Morty!'}"
    )
    parser.add_argument("action", type=BPEAction, choices=list(BPEAction))
    parser.add_argument(
        "--training_dataset",
    )
    parser.add_argument("--training_output")
    parser.add_argument("--vocabulary_size", type=int)
    parser.add_argument("--tokenizer_data")
    parser.add_argument("--run_data")
    parser.add_argument(
        "--in_place",
        help="Use carefully! Use it if you want to modify dataset file during training. It will corrupt your dataset, but save memory during training",
    )
    parser.add_argument(
        "text", nargs="?", help="Text to be tokenized when running the tokenizer"
    )
    args = parser.parse_args()

    if args.action == BPEAction.train:
        training_dataset = args.training_dataset or default_training_dataset_location

        if not os.path.exists(training_dataset):
            print("Please provide a training dataset")
            exit(1)

        training_output = args.training_output or default_training_output_location
        output_dir = os.path.dirname(training_output)

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        vocabulary_size = args.vocabulary_size or default_vocabulary_size

        tokenizer = BPETokenizer()
        tokenizer.train(
            dataset_path=training_dataset,
            in_place=args.in_place or False,
            vocabulary_size=vocabulary_size,
        )

        save_output = {"vocabulary": tokenizer.vocabulary}

        with open(training_output, "w") as output:
            json.dump(save_output, output, indent=4)

        print(
            f"Tokenizer trained successfully. Vocabulary is now stored in {training_output}"
        )

    if args.action == BPEAction.tokenize:
        tokenizer_data_location = args.tokenizer_data or default_inference_data_location

        if not os.path.exists(tokenizer_data_location):
            print("Please provide tokenizer data as a JSON that contains vocabulary")
            exit(1)

        run_data_location = args.run_data or default_tokenize_data_location

        if not os.path.exists(run_data_location):
            print("Please provide text to tokenize")
            exit(1)

        with open(run_data_location, "r") as run_data_file:
            inference_content = json.load(run_data_file)
            inference_content = inference_content.get("data", [])

        with open(tokenizer_data_location, "r") as tokenizer_data_file:
            tokenizer_data = json.load(tokenizer_data_file)

        vocabulary = tokenizer_data.get("vocabulary")

        tokenizer = BPETokenizer(vocabulary=vocabulary)
        print(tokenizer.tokenize(inference_content))

    if args.action == BPEAction.detokenize:
        tokenizer_data_location = args.tokenizer_data or default_inference_data_location

        if not os.path.exists(tokenizer_data_location):
            print("Please provide tokenizer data as a JSON that contains vocabulary")
            exit(1)

        run_data_location = args.run_data or default_detokenize_data_location

        if not os.path.exists(run_data_location):
            print("Please provide text to detokenize")
            exit(1)

        with open(run_data_location, "r") as run_data_file:
            inference_content = json.load(run_data_file)
            inference_content = inference_content.get("data", [])

        with open(tokenizer_data_location, "r") as tokenizer_data_file:
            tokenizer_data = json.load(tokenizer_data_file)

        vocabulary = tokenizer_data.get("vocabulary")

        tokenizer = BPETokenizer(vocabulary=vocabulary)
        print(tokenizer.detokenize(inference_content))
