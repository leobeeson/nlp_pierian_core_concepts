from ast import pattern
import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

matcher = Matcher(nlp.vocab)


pattern_1 = [{"LOWER": "solarpower"}]
pattern_2 = [{"LOWER": "solar"}, {"IS_PUNCT": True}, {"LOWER": "power"}]
pattern_3 = [{"LOWER": "solar"}, {"LOWER": "power"}]
matcher.add("solar_power", [pattern_1, pattern_2, pattern_3])

test_string_1 = "The Solar Power industry continues to grow as solarpower increases. Solar-power is amazing."
doc_1 = nlp(test_string_1)

found_matches = matcher(doc_1)
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc_1[start:end]
    print(match_id, string_id, start, end, span.text)

# Remove pattern from matcher:
matcher.remove("solar_power")
found_matches = matcher(doc_1)
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc_1[start:end]
    print(match_id, string_id, start, end, span.text)

# Add quantifiers to patterns:
pattern_1 = [{"LOWER": "solarpower"}]
pattern_2 = [{"LOWER": "solar"}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": "power"}]
matcher.add("solar_power", [pattern_1, pattern_2])

test_string_2 = "Solar--Power is solarpower yay!."
doc_2 = nlp(test_string_2)
found_matches = matcher(doc_2)
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc_2[start:end]
    print(match_id, string_id, start, end, span.text)

# Phrase matching:
phrase_matcher = PhraseMatcher(nlp.vocab)

with open("../data/TextFiles/reaganomics.txt",'r', encoding='latin1') as input_text:
    doc_3 = nlp(input_text.read())

phrases_list = [
    "voodoo economics", 
    "supply-side economics", 
    "trickle-down economics", 
    "free-market economics"
    ]

phrase_patterns = [nlp(text) for text in phrases_list]
type(phrase_patterns[0])

phrase_matcher.add("econ_matcher", phrase_patterns)
found_matches = phrase_matcher(doc_3)
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc_3[start:end]
    print(match_id, string_id, start, end, span.text)

