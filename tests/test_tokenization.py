import unittest

from tests.utils import initialize_pretrained_tokenizer


class TestTokenization(unittest.TestCase):
    def setUp(self):
        self.tokenizer = initialize_pretrained_tokenizer()

    def test_tokenize(self):
        tokens = self.tokenizer.tokenize(
            "Rick, you old bastard! Morty, you young bastard!"
        )
        print(tokens)
        self.assertEqual(
            tokens,
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
            ],
        )


if __name__ == "__main__":
    unittest.main()
