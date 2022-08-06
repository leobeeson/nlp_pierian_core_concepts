import spacy
from scipy import spatial
from numpy import ndarray

nlp = spacy.load("en_core_web_lg")

for s in nlp.vocab.vectors:
    _ = nlp.vocab[s]
type(_.vector)


def cosine_similarity(vec_1: ndarray, vec_2: ndarray) -> float:
    similarity = 1 - spatial.distance.cosine(vec_1, vec_2)
    return similarity
   
def compute_similarities(vector: ndarray) -> list[tuple[str, float]]:
    computed_similarities = []
    for word in nlp.vocab:
        if word.has_vector:
            if word.is_lower:
                if word.is_alpha:
                    similarity = cosine_similarity(vector, word.vector)
                    computed_similarities.append((word, similarity))
    return computed_similarities

def select_top_n_similar_words(computed_similarities: list[tuple[str, float]], n=10) -> list[str]:
    sorted_similarities = sorted(computed_similarities, key=lambda t:-t[1])
    top_words = [word[0].text for word in sorted_similarities[:n]]
    return top_words


def vector_math(word_a: str, word_b: str, word_c:str) -> list[str]:
    aggregate = nlp.vocab[word_a].vector - nlp.vocab[word_b].vector + nlp.vocab[word_c].vector
    computed_similarities = compute_similarities(aggregate)
    top_words = select_top_n_similar_words(computed_similarities)
    return top_words


vector_math("violence", "malice", "compassion")
vector_math("independence", "solitude", "community")
