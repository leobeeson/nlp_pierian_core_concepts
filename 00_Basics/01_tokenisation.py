from click import option
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

test_string = '"We\'re moving to L.A.!"'
doc =nlp(test_string)
for token in doc:
    print(token.text)


test_string_2 = "We're here to help! send snail-mail, email support@oursite.com or visit us a http://www.oursite.com!"
doc =nlp(test_string_2)
for token in doc:
    print(token.text)


test_string_3 = "A 5km NYC cab ride costs $10.30"
doc =nlp(test_string_3)
for token in doc:
    print(token.text)


test_string_4 = "Let's visit St. Louis in the U.S. next year."
doc =nlp(test_string_4)
for token in doc:
    print(token.text)
len(doc)
len(doc.vocab)

test_string_5 = "It is better to give than receive."
doc =nlp(test_string_5)
doc[0]
doc[2:5]

test_string_6 = "Apple to build a Hong Kong factory for $6 million."
doc = nlp(test_string_6)
for token in doc:
    print(token.text,end=" | ")

for entity in doc.ents:
    print(entity)
    print(entity.label_)
    print(str(spacy.explain(entity.label_)))
    print("\n")

test_string_7 = "Autonomous cars shift insurance liability towards manufacturers."
doc = nlp(test_string_7)
for chunk in doc.noun_chunks:
    print(chunk)


test_string_8 = "Apple is going to build a U.K. factory for $6 million."
doc = nlp(test_string_8)
displacy.render(doc, style="dep", jupyter=True, options={"distance": 110})


test_string_9 = "Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million."
doc = nlp(test_string_9)
displacy.render(doc, style="ent", jupyter=True, options={"distance": 110})
displacy.serve(doc, style="ent") # Need to fix connectivity from within WSL


