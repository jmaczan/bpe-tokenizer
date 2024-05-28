from io import TextIOWrapper


class BPETokenizer:
    def __init__(self, vocabulary: dict = {}, merge_rules: dict = {}) -> None:

        self.vocabulary = vocabulary
        self.merge_rules = merge_rules
        self.token_frequencies = {}

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
        """

        if dataset_path is None:
            raise Exception(
                "Please specify a path to a local file containing dataset, so it can be read in chunks into the tokenizer"
            )

        # first pass
        with open(self.dataset_path, "r", encoding="utf-8") as dataset_file:
            for chunk in self.read_chunk(dataset_file):
                self.tokenize_chunk()
                pass

        # next passes
        if in_place:
            working_copy_path = dataset_path
        else:
            working_copy_path = "working_copy.txt"
            self.duplicate_file(dataset_path, working_copy_path)

    def duplicate_file(
        source_path: str, destination_path: str, chunk_size: int = 1024 * 1024
    ):
        """
        Duplicates a potentially large file by reading and writing in chunks.

        :param source_path: Path to the source file
        :param destination_path: Path to the destination file
        :param chunk_size: Size of each chunk to read and write (default: 1MB)
        """
        with open(source_path, "rb") as source_file:
            with open(destination_path, "wb") as dest_file:
                while True:
                    chunk = source_file.read(chunk_size)
                    if not chunk:
                        break
                    dest_file.write(chunk)

    def read_chunk(self, file: TextIOWrapper, chunk_size: int = 512):
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

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
