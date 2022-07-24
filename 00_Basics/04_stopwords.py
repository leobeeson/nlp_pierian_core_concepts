import spacy

nlp = spacy.load("en_core_web_sm")

print(nlp.Defaults.stop_words)
len(nlp.Defaults.stop_words)

# Test if a word is in spacy's stopwords list:
nlp.vocab["is"].is_stop
nlp.vocab["mystery"].is_stop
nlp.vocab["btw"].is_stop
    
# Add stopwords to spacy's stopwords list:
nlp.Defaults.stop_words.add("btw")
nlp.vocab["btw"].is_stop = True
nlp.vocab["btw"].is_stop

# Remove stopwords to spacy's stopwords list:
nlp.Defaults.stop_words.remove("btw")
nlp.vocab["btw"].is_stop = False
nlp.vocab["btw"].is_stop
