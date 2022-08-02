import spacy
from scipy import spatial

nlp = spacy.load("en_core_web_lg")
len(nlp.vocab.vectors) # 514,157
nlp.vocab.vectors.shape # (514157, 300)


nlp("lion").vector
nlp("lion").vector.shape # (300, )

nlp("The quick brown fox jumped").vector
nlp("The quick brown fox jumped").vector.shape # (300, )


tokens = nlp("lion cat pet")
for token_1 in tokens:
    for token_2 in tokens:
        print(token_1.text, token_2.text, token_1.similarity(token_2))
# lion lion 1.0
# lion cat 0.3854507803916931
# lion pet 0.20031584799289703
# cat lion 0.3854507803916931
# cat cat 1.0
# cat pet 0.732966423034668
# pet lion 0.20031584799289703
# pet cat 0.732966423034668
# pet pet 1.0


tokens = nlp("like love hate")
for token_1 in tokens:
    for token_2 in tokens:
        print(token_1.text, token_2.text, token_1.similarity(token_2))
# Notice the three are very similar. It shows the vectors have learned the syntax, but not the semantics.
# This is because the model understands they're often used within the similar context.
# like like 1.0
# like love 0.5212638974189758
# like hate 0.5065140724182129
# love like 0.5212638974189758
# love love 1.0
# love hate 0.5708349943161011
# hate like 0.5065140724182129
# hate love 0.5708349943161011
# hate hate 1.0


tokens = nlp("dog cat nargle")
for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)


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
print([t[0].text for t in computed_similarities_sorted[:10]])


 