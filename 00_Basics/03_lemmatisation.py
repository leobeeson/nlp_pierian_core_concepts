import spacy

nlp = spacy.load("en_core_web_sm")

test_string_1 = "I am a runner running in a race because I love to run since I ran today."
doc_1 = nlp(test_string_1)
for token in doc_1:
    print(token.text, "\t", token.pos_, "\t", token.lemma, "\t", token.lemma_)

def show_lemmas(text: str):
    for token in text:
        print(f"{token.text:{12}} {token.pos_:{6}} {token.lemma:<{22}} {token.lemma_}")

show_lemmas(doc_1)


test_string_2 = "I saw 10 mice today!"
doc_2 = nlp(test_string_2)
show_lemmas(doc_2)

