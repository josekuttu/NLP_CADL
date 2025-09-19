# Install if not already installed:
# pip install spacy pandas
# python -m spacy download en_core_web_sm

import spacy
import pandas as pd

# Load pretrained spaCy NER model
nlp = spacy.load("en_core_web_sm")

# ------------------------------
# 1. Sample unstructured dataset (replace with your dataset)
# ------------------------------
documents = [
    "Elon Musk is the CEO of Tesla.",
    "Sundar Pichai works at Google in the United States.",
    "Bill Gates founded Microsoft Corporation.",
    "Satya Nadella is the CEO of Microsoft.",
    "Mark Zuckerberg created Facebook while studying at Harvard."
]

# ------------------------------
# 2. Run NER on each document
# ------------------------------
person_org_pairs = []

for text in documents:
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

    # Create combinations of persons and organizations in the same text
    for person in persons:
        for org in orgs:
            person_org_pairs.append((person, org))

# ------------------------------
# 3. Store structured data in DataFrame
# ------------------------------
df = pd.DataFrame(person_org_pairs, columns=["Person", "Organization"])
print(df)

# ------------------------------
# 4. Save to CSV for later use
# ------------------------------
df.to_csv("person_org_extraction.csv", index=False)
print("\nStructured information saved to person_org_extraction.csv")
