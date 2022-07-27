import spacy
from spacy.tokens.doc import Doc
from spacy.language import Language

nlp =spacy.load("en_core_web_sm")

sents_1 = "This is the first sentence. This is another sentence. This is the last sentence."
doc_1 = nlp(sents_1)
for sent in doc_1.sents:
    print(sent)

sents_2 = '"Management is doing the right things; leadership is doing the right things." -Peter Drucker'
doc_2 = nlp(sents_2)
doc_2.text
for sent in doc_2.sents:
    print(sent)
    print("\n")


# Add a segmentation rule:
## You need to restart the environment. 
# To add a custom component, the `nlp` object must be recently instantiated, i.e.
# you can't have used it before with a previous text.
import spacy
from spacy.tokens.doc import Doc
from spacy.language import Language

nlp = spacy.load("en_core_web_sm")

@Language.component("custom_boundaries")
def set_custom_boundaries(doc: Doc) -> Doc:
    for token in doc[:-1]:
        if token.text == ";":
            doc[token.i+1].is_sent_start = True
    return doc

nlp.add_pipe("custom_boundaries", name="split_colons", before="parser")
print(nlp.pipe_names) 
sents_2 = '"Management is doing the right things; leadership is doing the right things." -Peter Drucker'
doc_2 = nlp(sents_2)
for sent in doc_2.sents:
    print(sent)




