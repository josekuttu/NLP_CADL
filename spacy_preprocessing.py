import spacy

nlp = spacy.load("en_core_web_sm")

text = "Australian batting great Matthew Hayden and South African pace legend Dale Steyn have criticised Pat Cummins’ ‘defensive’ captaincy in the 2025 World Test Championship (WTC) final at Lord’s. They believe Australia allowed the game to slip away by not putting more pressure on Aiden Markram and Temba Bavuma — who was playing on one leg — when the Proteas were two wickets down."

doc = nlp(text)

print("sentence tokenization:")
for sent in doc.sents:
    print(sent)

print("Word tokenization: ")
for token in doc:
    print(token.text)

print("After stop word removal: ")
filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
print(filtered_tokens)

print("lemmatization:")
for token in doc:
    if not token.is_stop and not token.is_punct:
        print(f"{token.text} -> {token.lemma_}")