from click import option
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

# POS: universal part of speech -> UPOS: https://universaldependencies.org/u/pos/
# TAG: English part-of-speech tag -> OntoNotes 5: https://catalog.ldc.upenn.edu/LDC2013T19
# DEP: Clear dependency labels: https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md
# https://stackoverflow.com/a/40288324/7133282

string_1 = "The quick brown fox jumped over the lazy dog's back."
doc_1 = nlp(string_1)

print(doc_1.text)
print(doc_1[4].tag) #17109001835818727656
print(doc_1[4].tag_) #VBD (fine-grained tag)
print(doc_1[4].pos) #100
print(doc_1[4].pos_) #VERB 

for token in doc_1:
    print(f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_):{10}}")

string_2 = "I read books on NLP."
doc_2 = nlp(string_2)
word = doc_2[1]
word.text
token = word
print(f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_):{10}}")

string_3 = "Yesterday I read a book on NLP."
doc_3 = nlp(string_3)
word = doc_3[2]
word.text
token = word
print(f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_):{10}}")



pos_counts = doc_1.count_by(spacy.attrs.POS)
for k,v in sorted(pos_counts.items()):
    print(f"{k}. {doc_1.vocab[k].text:{5}} {v}")

tag_counts = doc_1.count_by(spacy.attrs.TAG)
for k,v in sorted(tag_counts.items()):
    print(f"{k}. {doc_1.vocab[k].text:{5}} {v}")

dep_counts = doc_1.count_by(spacy.attrs.DEP)
for k,v in sorted(dep_counts.items()):
    print(f"{k}. {doc_1.vocab[k].text:{5}} {v}")


displacy.render(doc_1, style="dep", jupyter=True)

options = {"distance": 110, "compact": True, "color": "yellow", "bg": "#09a3d5", "font": "Times"}
displacy.render(doc_1, style="dep", jupyter=True, options=options)

# Multiple sentences, manage as spans (list of sentences):
string_4 = "This is a sentence. This is another sentence, possible longer than the first sentence."
doc_4 = nlp(string_4)
spans_4 = list(doc_4.sents)
displacy.render(spans_4, style="dep", jupyter=True, options={"distance": 110})
