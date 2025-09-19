# -------------------------------
# Install required libraries (uncomment if needed)
# -------------------------------
# !pip install gensim nltk spacy pyLDAvis

# -------------------------------
# Imports
# -------------------------------
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import nltk
import pyLDAvis
import pyLDAvis.gensim_models

# -------------------------------
# Step 0: Define Text Corpus
# -------------------------------
corpus = [
    "Artificial intelligence (AI) is transforming industries across the globe. Companies are using AI for predictive analytics, natural language processing, and automation to improve efficiency and reduce costs.",
    "Machine learning, a subset of AI, allows systems to learn from data and improve performance over time. Applications include recommendation systems, image recognition, and autonomous vehicles.",
    "Climate change continues to be a pressing global issue. Rising temperatures, melting glaciers, and extreme weather events are affecting ecosystems, agriculture, and human populations.",
    "Renewable energy sources, such as solar, wind, and hydropower, are essential to reduce carbon emissions and combat climate change. Governments and businesses are investing in green technologies.",
    "The stock market is influenced by various economic indicators, including inflation rates, employment data, and corporate earnings. Investor sentiment also plays a crucial role in market fluctuations.",
    "Financial literacy is important for individuals to make informed decisions about savings, investments, and debt management. Personal finance education can lead to better economic outcomes.",
    "Healthcare systems worldwide are adopting digital technologies, such as electronic health records and telemedicine, to improve patient care. AI is also being used for diagnostics and personalized treatment.",
    "Nutrition and exercise are fundamental for maintaining good health. Public awareness campaigns encourage balanced diets, regular physical activity, and preventive health measures.",
    "Education is evolving with the integration of technology. Online learning platforms, digital textbooks, and virtual classrooms are transforming how students access knowledge and skills.",
    "Research in neuroscience is uncovering how the human brain processes information. Advances in brain imaging and computational modeling are contributing to understanding cognition and behavior.",
    "Cybersecurity has become critical as more personal and business data are stored online. Organizations invest in security protocols, encryption, and training to protect against data breaches and cyberattacks.",
    "Space exploration continues to advance with missions to Mars, asteroid mining prospects, and the development of reusable rockets. Private companies are increasingly playing a role in the space industry.",
    "Urbanization is changing the dynamics of cities. Smart city initiatives focus on improving infrastructure, reducing traffic congestion, and enhancing public services through technology.",
    "Mental health awareness is growing globally. Access to counseling, therapy apps, and community support programs is helping individuals cope with stress, anxiety, and depression.",
    "Agricultural technology, including precision farming and drone monitoring, is increasing crop yields and resource efficiency. Sustainable practices are crucial for food security and environmental preservation."
]

# -------------------------------
# Step 1: Preprocessing
# -------------------------------
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load spacy model
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove punctuation and stopwords
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Lemmatization
    doc = nlp(" ".join(tokens))
    tokens = [token.lemma_ for token in doc]
    return tokens

processed_corpus = [preprocess(doc) for doc in corpus]

# -------------------------------
# Step 2: Dictionary & Corpus for LDA
# -------------------------------
dictionary = corpora.Dictionary(processed_corpus)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_corpus]

# -------------------------------
# Step 3: LDA Model
# -------------------------------
num_topics = 5  # Change number of topics as needed

lda_model = LdaModel(
    corpus=doc_term_matrix,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=42,
    passes=15,
    alpha='auto',
    per_word_topics=True
)

# Print topics
print("\nLDA Topics:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx+1}: {topic}")

# -------------------------------
# Step 4: Visualize Topics and Save as HTML
# -------------------------------
lda_vis = pyLDAvis.gensim_models.prepare(lda_model, doc_term_matrix, dictionary)

# Save visualization as HTML
pyLDAvis.save_html(lda_vis, 'lda_visualization.html')
print("\nVisualization saved as 'lda_visualization.html'. Open it in your browser to view it.")
