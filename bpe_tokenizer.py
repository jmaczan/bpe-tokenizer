from collections import defaultdict
from enum import Enum
import json
import tempfile
from bpe_utils import dict_to_defaultdict, duplicate_file, read_chunk
import os
import argparse


class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer
    """

    def __init__(self, vocabulary: dict = None, merge_rules: dict = None) -> None:

        self.vocabulary = dict_to_defaultdict(
            vocabulary or {token: token for token in range(256)}
        )
        self.merge_rules = merge_rules or defaultdict(int)
        self.token_frequencies = defaultdict(int)

    def train(
        self,
        dataset_path,
        output_path: str = None,
        vocabulary_size: int = 50_257,
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
        - Replace all occurences of the most frequent pair in dataset (working copy or original, if in_place=True)
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

        while len(self.vocabulary) < vocabulary_size and (
            len(self.token_frequencies) == 0
            or not sorted(self.token_frequencies)[0][1][1] == 1
        ):
            self.token_frequencies = defaultdict(int)  # reset token frequencies counter
            with open(
                working_copy_path, "r", encoding="utf-8"
            ) as source_copy:  # maybe encoding here and decoding later is redundant?
                for chunk in read_chunk(source_copy):
                    self.count_token_frequencies(chunk.encode("utf-8"))

            most_frequent_pair = sorted(set(self.token_frequencies.items()))[0]

            new_token_index = len(self.vocabulary)
            self.vocabulary[new_token_index] = (
                most_frequent_pair[0][0] + most_frequent_pair[0][1]
            )  # it might be wrong method of concatenating bytes

            temp_file, temp_file_path = tempfile.mkstemp()

            # replace all occurences of a most frequent pair
            with open(
                working_copy_path,
                "a+",
                encoding="utf-8",  # maybe encoding here and decoding later is redundant?
            ) as source_copy, open(temp_file, "wb", encoding="utf-8") as temp_copy:
                for chunk in read_chunk(source_copy):
                    chunk = chunk.encode("utf-8")
                    for token in len(chunk) - 1:
                        if (
                            most_frequent_pair[0] == chunk[token]
                            and most_frequent_pair[1] == chunk[token + 1]
                        ):
                            chunk = chunk[:token] + new_token_index + chunk[token + 2 :]

                        temp_copy.write(chunk)

            os.replace(temp_file_path, working_copy_path)

    def count_token_frequencies(self, data):
        for index in range(0, len(data), 2):
            self.token_frequencies[(data[index], data[index + 1])] += 1

    def run(self):
        pass

    def load_vocabulary(self, vocabulary: dict = {}):
        self.vocabulary = vocabulary

    def load_merge_rules(self, merge_rules: dict = {}):
        self.merge_rules = merge_rules

    def load(self, vocabulary: dict = {}, merge_rules: dict = {}):
        self.merge_rules = merge_rules
        self.vocabulary = vocabulary


class BPEAction(Enum):
    train = "train"
    run = "run"

    def __str__(self) -> str:
        return self.value


default_training_dataset_location = "training.txt"
default_training_output_location = "tokenizer.json"
default_inference_data_location = default_training_output_location

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Byte-Pair Encoding Tokenizer. Default training dataset location: 'training.txt'. Default inference (run) data location 'tokenizer.txt' - it includes JSONs with vocabulary and merge rules."
    )
    parser.add_argument("action", type="BPEAction", choices=list(BPEAction))
    parser.add_argument(
        "--training_dataset",
    )
    parser.add_argument("--training_output")
    parser.add_argument("--tokenizer_data")
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

        tokenizer = BPETokenizer()
        vocabulary, merge_rules = tokenizer.train()

        save_output = {"vocabulary": vocabulary, "merge_rules": merge_rules}

        with open(training_output, "w") as output:
            json.dump(save_output, output, indent=4)

        print(
            f"Tokenizer trained successfully. Vocabualry and merge rules are now stored in {training_output}"
        )

    if args.action == BPEAction.run:
        tokenizer_data_location = args.tokenizer_data or default_inference_data_location

        if not os.path.exists(tokenizer_data_location):
            print(
                "Please provide tokenizer data as a JSON that contains vocabulary and merge_rules"
            )
            exit(1)

        tokenizer = BPETokenizer()
        print(tokenizer.run())
