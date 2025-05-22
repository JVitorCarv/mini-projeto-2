import json
import nltk
import os
import re
import string
import torch
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer


class NLPPreprocessor:
    def __init__(self, vocab_path: str):
        self._load_nltk_pkgs()

        self.vocab_path = vocab_path

        self.vocab = self._load_vocab()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = TreebankWordTokenizer()

        self.raw_text = None
        self.stripped_text = None
        self.normalized_text = None
        self.indexed_text = None
        self.tensor = None

    @staticmethod
    def _load_nltk_pkgs():
        """
        Download NLTK packages if not already present.
        """
        for pkg in ["punkt", "wordnet", "stopwords", "omw-1.4"]:
            try:
                nltk.data.find(pkg)
            except LookupError:
                nltk.download(pkg, quiet=True)

    @staticmethod
    def _load_vocab(path: str = os.path.join("nlp", "vocab.json")) -> dict[str, int]:
        """
        Load the JSON file created during training.
        Returns:
            dict: Vocabulary mapping words to indices.
        """
        with open(path, encoding="utf-8") as f:
            vocab = json.load(f)
        if "<unk>" not in vocab or "<pad>" not in vocab:
            raise ValueError("Vocabulary must contain <pad> and <unk>")
        return vocab

    def _strip_text(self):
        """
        Remove unwanted characters from the text.

        Returns:
            str: Cleaned text.
        """
        text = self.raw_text
        text = text.replace("\\n", " ").replace("\n", " ")  # remove newlines
        text = text.lower()
        text = re.sub(r"http\S+", "", text)  # remove URLs
        text = re.sub(r"@\w+", "", text)  # remove mentions
        text = re.sub(r"#\w+", "", text)  # remove hashtags
        text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
        text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace

        self.stripped_text = text
        return text

    def _normalize_text(self):
        """
        Normalize the text by removing punctuation and lemmatizing.

        Returns:
            str: Normalized text.
        """
        words = self.tokenizer.tokenize(self.stripped_text)
        cleaned = [
            self.lemmatizer.lemmatize(w)
            for w in words
            if w.lower() not in self.stop_words and w not in string.punctuation
        ]
        self.normalized_text = " ".join(cleaned)
        return self.normalized_text

    def _index_text(self):
        """
        Convert the normalized text into a list of vocabulary indices.

        Returns:
            List[int]: The sequence of integer token IDs corresponding to the words.
            Tokens not present in the vocabulary are replaced by the <unk> index.
        """
        self.indexed_text = [
            self.vocab.get(word, self.vocab["<unk>"])
            for word in self.normalized_text.split()
        ]
        return self.indexed_text

    def _to_tensor(
        self, max_len: int = 120, pad_idx: int | None = None
    ) -> torch.Tensor:
        """
        Return a 1 * max_len LongTensor, padded/truncated to max_len.
        """
        if pad_idx is None:
            pad_idx = self.vocab.get("<pad>", 0)
        idx = self.indexed_text
        seq = idx[:max_len] + [pad_idx] * (max_len - len(idx))
        self.tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
        return self.tensor

    def preprocess_text(self, text: str) -> torch.Tensor:
        """
        Preprocess the input text and return a tensor representation.
        """
        self.raw_text = text
        self._strip_text()
        self._normalize_text()
        self._index_text()
        return self._to_tensor()


if __name__ == "__main__":
    pass
