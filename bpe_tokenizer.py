from io import TextIOWrapper
from collections import defaultdict

from bpe_utils import dict_to_defaultdict, duplicate_file, read_chunk


class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer

    It uses bytes internally, so any character consists of n bytes from range (0, 255), where n>0
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

        Procedure:
        We do things in chunks because datasets might be huge so we don't want to load all the dataset into memory at once
        - Count a frequency of any two subsequent tokens, like in a sentence "hey, hello" we would identify (("h", "e"), 2) and all other pairs would have 1 occurence
        - Pick a pair that has the most occurences
        - Combine those tokens and put it into vocabulary
        - Replace all occurences of the most frequent pair in dataset (working copy or original, if in_place=True)
        - Reset token frequencies object
        - Repeat from first step until we get a vocabulary of length of vocabulary_size or when all pairs have only 1 occurence

        :param in_place: if True, then will modify the dataset when training a tokenizer, so it saves a disk space by not copying a dataset during training (bool)
        """

        if dataset_path is None:
            raise Exception(
                "Please specify a path to a local file containing dataset, so it can be read in chunks into the tokenizer"
            )

        # first pass
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
            with open(self.dataset_path, "r", encoding="utf-8") as working_copy_path:
                for chunk in read_chunk(working_copy_path):
                    self.count_token_frequencies(chunk.encode("utf-8"))

            most_frequent_pair = sorted(set(self.token_frequencies.items()))[0]
            most_frequent_pair_concatenated = (
                most_frequent_pair[0][0]
                + most_frequent_pair[0][1]  # likely this concatenation way is wrong
            )

            self.vocabulary[len(self.vocabulary)] = most_frequent_pair_concatenated

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


if __name__ == "__main__":
    tokenizer = BPETokenizer()
    vocabulary, merge_rules = tokenizer.train()
