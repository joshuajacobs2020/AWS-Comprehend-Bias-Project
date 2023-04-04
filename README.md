QTM-350 - Group-4
====

[README.md](https://github.com/joshuajacobs2020/QTM-350---Group-4/files/11152296/README.md)

## Using Comprehend to analyze the sentiment of text

For this walkthrough, we import **boto3**  to be able to use the various methods in the Comprehend API.


```python
# Importing Packages
import boto3
import sagemaker
import json
import os
client = boto3.client('comprehend')
```

We create a function called sentiment test to be able to analyze the sentiment of a certain phrase. In this function, we  have two parameters  **entity**  and  **option**. For this project, we want  to see how changing the entity might affect the sentiment score.


```python
def sentiment_test(entity, option="overall"):
    body = "claims they want to protect children from the dangers of social media"
    text = f"{entity} {body}"
    respond = 0

    response = client.batch_detect_sentiment(
    TextList=[text],
    LanguageCode='en')
    
    if option == "overall":
        respond = response['ResultList'][0]['Sentiment']
    elif option == "positive":
        respond = response['ResultList'][0]['SentimentScore']['Positive']
    elif option == "negative":
        respond = response['ResultList'][0]['SentimentScore']['Negative']
    elif option == "mixed":
        respond = response['ResultList'][0]['SentimentScore']['Mixed']
    elif option == "neutral":
        respond = response['ResultList'][0]['SentimentScore']['Neutral']
    return respond
```

As you can see here, by inputting the entity as "Joe Biden" the negative sentiment score is 0.08


```python
sentiment_test("Joe Biden", option="negative")
```




    0.08307952433824539



In contrast, the negative sentiment score when inputting the entity as "Donald Trump" is 0.30


```python
sentiment_test("Donald Trump", option = "negative")
```




    0.3071601092815399



Although we create a python function to analyze the sentiment of the phrase by changing the entity, for our machine learning project we will input our own data to determine whether certain entities and their political affilation can affect their certain sentiment score. Furthermore, we will do a regression analysis to determine how political affilation and gender will change sentiment score.

We can use  `AWS CLI` commands in order to see another example of how Amazon Comprehend detects the sentiment for these two different entities. We input an actual [headline](https://www.politifact.com/article/2019/apr/26/context-trumps-very-fine-people-both-sides-remarks/) and just change the entity from Donald Trump to Joe Biden to see the different sentiment for both.


```python
!aws comprehend detect-sentiment --text "Donald Trump says there are fine people on both sides" --language-code en
```

    {
        "Sentiment": "NEUTRAL",
        "SentimentScore": {
            "Positive": 0.14906999468803406,
            "Negative": 0.22132088243961334,
            "Neutral": 0.583514392375946,
            "Mixed": 0.0460946261882782
        }
    }



```python
!aws comprehend batch-detect-sentiment --text "Joe Biden says there are fine people on both sides" --language-code en
```

    {
        "ResultList": [
            {
                "Index": 0,
                "Sentiment": "NEUTRAL",
                "SentimentScore": {
                    "Positive": 0.36996933817863464,
                    "Negative": 0.06190203130245209,
                    "Neutral": 0.5238028764724731,
                    "Mixed": 0.04432568699121475
                }
            }
        ],
        "ErrorList": []
    }


Although the overall sentiment score for both is considered "neutral", the entity that has "Joe Biden" has a higher positive sentiment score than "Donald Trump".
