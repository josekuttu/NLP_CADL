import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


text = "Australian batting great Matthew Hayden and South African pace legend Dale Steyn have criticised Pat Cummins’ ‘defensive’ captaincy in the 2025 World Test Championship (WTC) final at Lord’s. They believe Australia allowed the game to slip away by not putting more pressure on Aiden Markram and Temba Bavuma — who was playing on one leg — when the Proteas were two wickets down."


#sentence tokenization
print("Sentence Tokenization: ")
sentences = sent_tokenize(text)
for s in sentences:
  print(s)

#word tokenization
print("\nWord Tokenization: ")
words = word_tokenize(text)
print(words)

#stop word removal
print("\n After Stop Word Removal:")
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print(filtered_words)

#stemming
print("\nStemming : ")
stemmer = PorterStemmer()
for word in filtered_words:
  print(f"{word} -> {stemmer.stem(word)}")

#Lemmatization
print("\n Lemmation:")
lemmatizer = WordNetLemmatizer()
for word in filtered_words:
  print(f"{word} -> {lemmatizer.lemmatize(word)}")