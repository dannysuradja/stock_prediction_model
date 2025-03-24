import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import pipeline
from datetime import datetime, timedelta
from tqdm import tqdm


finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", device=0)

def get_finbert_score(article):
    finbert_score = finbert(article, truncation=True, max_length=512)
    prediction = finbert_score[0]
    
    # Convert the label to lowercase for consistent comparison
    label = prediction['label'].lower()
    score = prediction['score']
    
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:
        return 0

# Ge 3-day rolling average of finbert sentiment scores for each data (Output: Date // finbert_sentiment // rolling_mean_score)
def get_sentiment_table(dataset="sabareesh88/FNSPID_nasdaq"):
    news_dataset = load_dataset(dataset)
    news_dataset = news_dataset['train'].to_pandas()

    apple_news2 = news_dataset[news_dataset['Article'].str.contains("Apple")]
    reduced_apple_news2 = apple_news2[["Date","Article"]]

    reduced_apple_news2["Date"] = pd.to_datetime(reduced_apple_news2["Date"].astype(str).str.replace(" UTC", ""))
    reduced_apple_news2 = reduced_apple_news2[(reduced_apple_news2['Date'].dt.year >= 2014) & 
                                          (reduced_apple_news2['Date'].dt.year <= 2023)]
    
    # Apply get_finbert_score function for each article. (* This may take a while)
    tqdm.pandas()
    reduced_apple_news2['finbert_sentiment'] = reduced_apple_news2['Article'].progress_apply(get_finbert_score)
    reduced_apple_news2["Date"] = reduced_apple_news2["Date"].dt.strftime("%Y-%m-%d")

    # Assign mean finbert score for each date within time period
    grouped_score = reduced_apple_news2.groupby("Date")["finbert_sentiment"].mean().reset_index()
    grouped_score["Date"] = pd.to_datetime(grouped_score["Date"])

    full_date_range = pd.date_range(start=min(grouped_score["Date"]), end=max(grouped_score["Date"]))
    full_dates_df = pd.DataFrame({"Date": full_date_range})
    joined_df = pd.merge(full_dates_df, grouped_score, on="Date", how="left")
    joined_df = joined_df.fillna(0)

    # Assign 3 day rolling average score
    joined_df["rolling_mean_score"] = joined_df["finbert_sentiment"].rolling(window=5, min_periods=1).mean()
    joined_df.to_csv('sentiment_score', index=False)
    return joined_df


# References
# @misc{dong2024fnspid,
#   title={FNSPID: A Comprehensive Financial News Dataset in Time Series},
#   author={Zihan Dong and Xinyu Fan and Zhiyuan Peng},
#   year={2024},
#   eprint={2402.06698},
#   archivePrefix={arXiv},
#   primaryClass={q-fin.ST}
# }