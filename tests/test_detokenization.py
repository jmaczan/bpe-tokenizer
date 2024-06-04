import unittest
from tests.utils import initialize_pretrained_tokenizer


class TestDetokenization(unittest.TestCase):
    def setUp(self):
        self.tokenizer = initialize_pretrained_tokenizer()

    def test_detokenize(self):
        tokens = self.tokenizer.detokenize(
            [
                275,
                265,
                282,
                32,
                111,
                108,
                100,
                32,
                98,
                97,
                115,
                116,
                97,
                114,
                100,
                284,
                268,
                265,
                282,
                32,
                282,
                110,
                292,
                98,
                97,
                115,
                116,
                97,
                114,
                100,
                33,
            ]
        )
        self.assertEqual(tokens, "Rick, you old bastard! Morty, you young bastard!")


if __name__ == "__main__":
    unittest.main()
