import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

# Sample text corpus
text = """After hectic negotiations over the no-handshake controversy in Sunday’s India-Pakistan Asia Cup match, where the latter demanded an apology from match referee Andy Pycroft and an inquiry into his conduct at the toss, the country’s top cricketing brass addressed a press conference to clear their stand on the matter. They said that the prospect of pulling out of the tournament had also been on the table. Here’s what PCB chief Mohsin Naqvi, one of his predecessors Najam Sethi and former Pakistan captain Rameez Raja had to say:

Boycott was a very big decision: Mohsin Naqvi
As you all know, there has been a crisis going on since 14th September. We had objections about the role of the match referee (Andy Pycroft). Just a short while back, the match referee had a conversation with the team coach, captain and manager. He said that this incident (no handshakes) should not have happened. We had also requested the ICC earlier to set up an inquiry on the code violation during the match. We believe that politics and sports can’t go together. This is sports, and let it remain a sport. Cricket should be separate from all this.

I requested Sethi saab and Rameez Raja saab. If we had to go for a boycott, which was a very big decision – the prime minister, government officials and lots of other people were also involved, and we got their full support. We were monitoring the issue."""
doc = nlp(text)

# Tokenization
tokens = [token.text for token in doc]
print("Tokens:", tokens)

# Stopword removal
filtered_tokens = [token.text for token in doc if not token.is_stop]
print("After Stopword Removal:", filtered_tokens)

# Lemmatization
lemmas = [token.lemma_ for token in doc if not token.is_stop]
print("After Lemmatization:", lemmas)

# (SpaCy does not have a built-in stemmer, but lemmatization is usually preferred)
