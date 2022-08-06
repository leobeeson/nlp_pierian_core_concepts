import spacy
from scipy import spatial

nlp = spacy.load("en_core_web_lg")

# Do this in order to load from the model all string representations (lexemes) with vectors:
for s in nlp.vocab.vectors:
    _ = nlp.vocab[s]

cosine_similarity = lambda vec_1, vec_2: 1 - spatial.distance.cosine(vec_1, vec_2)
king = nlp.vocab["king"].vector
man = nlp.vocab["man"].vector
woman = nlp.vocab["woman"].vector
new_vector = king - man + woman

computed_similarities = []
for word in nlp.vocab:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_vector, word.vector)
                computed_similarities.append((word, similarity))
len(computed_similarities)

computed_similarities_sorted = sorted(computed_similarities, key=lambda item:-item[1])
print([t[0].text for t in computed_similarities_sorted[:20]])

