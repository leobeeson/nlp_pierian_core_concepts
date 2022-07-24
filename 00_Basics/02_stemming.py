import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer


words = ["run", "runner", "ran", "runs", "easily", "fairly", "fairness"]

p_stemmer = PorterStemmer()
for word in words:
    print(f"{word}----->{p_stemmer.stem(word)}")

s_stemmer = SnowballStemmer(language="english")
for word in words:
    print(f"{word}----->{s_stemmer.stem(word)}")

words_2 = ["generous", "generation", "generously", "generate"]
for word in words_2:
    print(f"{word}----->{s_stemmer.stem(word)}")
