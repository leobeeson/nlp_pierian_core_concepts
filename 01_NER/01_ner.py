import spacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")

def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(f"{ent.text} - {ent.label_} - {spacy.explain(ent.label_)}")
    else:
        print("No entities found")


doc_1 = nlp("Hi, how are you?")
show_ents(doc_1)


doc_2 = nlp("May I go to Washington, DC next May to see the Washington Monument?")
show_ents(doc_2)

doc_3 = nlp("Tesla to build a U.K. factory for $6 million.")
show_ents(doc_3)

ORG = doc.vocab.strings["ORG"]
new_ent = Span(doc_3, 0, 1, label=ORG)
doc_3.ents = list(doc_3.ents) + [new_ent]
show_ents(doc_3)


doc_4 = nlp("Our company created a brand new vacuum cleaner."
            "This new vacuum-cleaner is the best in show.") 
show_ents(doc_4)


phrase_matcher = PhraseMatcher(nlp.vocab)
phrase_list = ["vacuum cleaner", "vacuum-cleaner"]
phrase_patterns = [nlp(text) for text in phrase_list]
phrase_matcher.add("new_product", phrase_patterns)
found_matches = phrase_matcher(doc_4)

PROD = doc_4.vocab.strings["PRODUCT"]
new_ents = [Span(doc_4, match[1], match[2], label=PROD) for match in found_matches]
doc_4.ents = list(doc_4.ents) + new_ents
show_ents(doc_4)

# Filter occurrences of a specific type of entity, or count occurrences of a specific type:
doc_5 = nlp("Originally I paid $29.95 for this car toy, but not it is marked down by 10 dollars.")
[ent for ent in doc_5.ents if ent.label_ == "MONEY"]
len([ent for ent in doc_5.ents if ent.label_ == "MONEY"])
