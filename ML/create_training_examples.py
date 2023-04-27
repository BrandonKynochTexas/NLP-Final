from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from stonks_dataset_prep import ArticleCollector, Sentiment_Article
from dataclasses import dataclass, field
from typing import List
from datetime import datetime
from bs4 import BeautifulSoup
import pickle


"""
Dataclass for the training examples
"""

# Positive, Neutral, Negative
def default_sentiment_scores() -> List[float]:
    return [0, 0 ,0]

@dataclass
class TrainingExample:
    date: str
    price: float
    sentiment_scores: List[float] = field(default_factory=default_sentiment_scores)


def timestamp_to_dateime(timestamp):
    return datetime.fromtimestamp(int(timestamp))

# Prepare text for the model
def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()
    clean_text_abbr = clean_text[0:512]
    return clean_text_abbr

def create_finbert_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# calculates the article scores
def calc_article_score(text, model_pipeline):
    return model_pipeline(clean_text(text))

def create_training_data(articles, real_price_data):
    # Model pipeline for score inference
    model_pipeline = create_finbert_pipeline()

    # Price data to article alignment
    training_examples = []

    for idx, price_data in enumerate(real_price_data):
        price = price_data[0][1] # real_price_data[i][0][1] is close price
        current_date = timestamp_to_dateime(price_data[0][0]).date()

        if idx == 0:
            example = TrainingExample(current_date, 0)
        else:
            prev_price = real_price_data[idx - 1][0][1]
            percentage_difference = (price - prev_price) / prev_price * 100
            example = TrainingExample(current_date, percentage_difference)

        print(f"Sentiment scores for date: {current_date}")

        article_scores = []

        # Look for articles that match the same date
        for article in articles:
            if current_date == article.date.date():
                
                print(f"Working on article: {article.title} {article.date}")

                article_score = calc_article_score(article.body, model_pipeline)
                article_scores.append(article_score)
        
        if len(article_scores) > 0:
            positive_score = 0.0
            neutral_score = 0.0
            negative_score = 0.0
            for article_score in article_scores:
                
                # need to do this to because dictionary is behind a single item list
                indexed_score = article_score[0]

                label = indexed_score['label']
                score = indexed_score['score']

                if label == 'positive':
                    positive_score += score
                elif label == 'neutral':
                    neutral_score += score
                elif label == 'negative':
                    negative_score += score
                else:
                    raise Exception('Sentiment label not found')
                
            sentiment_scores = [positive_score, neutral_score, negative_score]
            sscore_norm = [score_x / len(article_scores) for score_x in sentiment_scores]
            example.sentiment_scores = sscore_norm
        
        training_examples.append(example)
    return training_examples


if __name__== "__main__":

    TICKER = "AMD" # PUT TICKER HERE

    # TODO: Change this to be dynamic and configurable
    with open(f'./Dataset/{TICKER}-article-collector', 'rb') as f:
        finviz = pickle.load(f).articles
        
    with open(f'./Dataset/{TICKER}-motley-article-collector', 'rb') as f:
        motley = pickle.load(f).articles

    combined_articles = finviz.copy()
    combined_articles.extend(motley)

    earliest_article_date = combined_articles[0].date.date()

    for article in combined_articles:
        if article.date.date() < earliest_article_date:
            earliest_article_date = article.date.date()

    earliest_price_idx = float('inf')

    with open(f'./Dataset/{TICKER}-price-data-Interval.in_daily', 'rb') as f:
        price_data = pickle.load(f)

    # Find the index of the price data of the earliest article
    for idx, date in enumerate(price_data):
        current_date = timestamp_to_dateime(date[0][0]).date()
        if  earliest_article_date >= current_date :
            earliest_price_idx = idx
    
    print("Earliest article date:", earliest_article_date)
    print("Total Articles: ", len(combined_articles))

    if earliest_price_idx == float('inf'):
        raise Exception('Earliest article date preceeds earliest price data')

    real_price_data = price_data[earliest_price_idx:]
    training_examples = create_training_data(combined_articles, real_price_data)

    pickle.dump(training_examples, open(f'./training_examples/{TICKER}_training_example.pkl', 'wb'))


