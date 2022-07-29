import os


# Build a simple vocab:
class Vocab:

    def __init__(self) -> None:
        self.vocab = {}
        self.vocab_idx = 1

    def _add_tokens_to_vocab(self, tokens: list[str]) -> None:
        for token in tokens:
            if token in self.vocab:
                continue
            self.vocab[token] = self.vocab_idx
            self.vocab_idx += 1
    
    def _read_file_into_lowercase_list(self, filepath: str) -> list[str]:
        with open(filepath) as input_file:
            tokenised_text = input_file.read().lower().split()
        return tokenised_text

    def build_vocab(self, corpus_files: list[str]) -> None:
        for corpus_file in corpus_files:
            tokenised_text = self._read_file_into_lowercase_list(corpus_file)
            self._add_tokens_to_vocab(tokenised_text)

    def _vectorise_document(self, filepath: str) -> list[str]:
        vectorised_doc = [filepath] + [0]*len(self.vocab)
        with open(filepath) as input_file:
            tokenised_text = input_file.read().lower().split()
        for token in tokenised_text:
            vectorised_doc[self.vocab[token]] += 1
        return vectorised_doc


if __name__ == "__main__":
    os.chdir("./03_text_classification")    
    vocab = Vocab()
    print(f"Vocab at instantiation:\n{vocab.vocab}")

    corpus_files = [
        "../data/03-Text-Classification/1.txt",
        "../data/03-Text-Classification/2.txt"
    ]
    vocab.build_vocab(corpus_files)
    print(f"Populated vocab:\n{vocab.vocab}")

    for corpus_file in corpus_files:
        print(vocab._vectorise_document(corpus_file))
    