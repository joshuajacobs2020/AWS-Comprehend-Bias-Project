## AWS Comprehend Sentiment Analysis


The **purpose** of this project is to gauge whether or not AWS Comprehend has any sort of political bias by changing the entity given a body of text.



### Research

A comprehensive literature review was conducted  in order to throughly investigate past research on if NLP models have political bias
* Background knowledge can be found [here](https://mlwebsitepoliticalbias.s3.amazonaws.com/HTMLBlog.html)

## Project Description

The purpose of this project was to describe how to use Comprehend API specifically using `entity recognition` and  `sentiment analysis`.  AWS provides a Natural Language Processing (NLP) service that can be used to analyze a body of text.  Our project revolves around creating a python function using the the `batch_detect_sentiment method` which can give a sentiment score for the corresponding entity. A table has been provided to demonstrate what the results might look like.


| Entity | Type | Text| Sentiment |
| -------- | -------- | -------- | -------- |
| Ron DeSantis | `person`|  promotes human rights for marginalized groups  | Neutral |
| Kamala Harris  | `person` | promotes human rights for marginalized groups| Negative |
| Vladimir Putin | `person` | promotes human rights for marginalized groups | Positive |
| Xi JinPing | `person` | promotes human rights for marginalized groups | Neutral|

* This table is not indicative of the actual sentiment from AWS Comprehend


There are a variety of users who may want to reproduce this work if they want to conduct a sentiment analysis for different entities.

1. Students:  Students can use this project to learn new skills relating to AWS services.
2. Journalists: Journalists may want to gauge whether the heading of their article has some sort of political bias.
3. Developers: Developers may want to understand how to use Amazon Comprehend API and see a specific use case especially if it relates to sentiment analysis
4. Politicians: Politicians can utilize the results from the code provided in order to understand their sentiment from an AI perspective.


### Reproducibility Guide

**In order to understand how to use these files, a detailed description is given for the files needed to work and run this code**

1. The User should start off reading some background information about the inherent political bias in NLP models. The blog post also includes a step-by-step process of how we analyzed text for people in Congress. 
[Blog.md](https://github.com/joshuajacobs2020/QTM-350---Group-4/blob/main/Blog.md)

2. Visiting  [ComprehendGuide.ipynb](https://github.com/joshuajacobs2020/QTM-350---Group-4/blob/main/ComprehendGuide.ipynb)  next can help guide the user to see how to initialize boto3 and provides an example of a function where one can change the text and entity to see sentiment score for a specific entity. The function is provided here:

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
 
3. The  [DataCreation.ipynb](https://github.com/joshuajacobs2020/QTM-350---Group-4/blob/main/DataCreation.ipynb) file gives an in-depth guide on how to clean the dataset and generate sentiment scores for a bunch of different entities with different phrases. Feel free to use your own dataset that have other characteristics for the independent variable one wants to analyze.


```python
import pandas as pd
df = pd.read_csv('QTM-350---Group-4/Congress.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>sort_name</th>
      <th>email</th>
      <th>twitter</th>
      <th>facebook</th>
      <th>group</th>
      <th>group_id</th>
      <th>area_id</th>
      <th>area</th>
      <th>chamber</th>
      <th>term</th>
      <th>start_date</th>
      <th>end_date</th>
      <th>image</th>
      <th>gender</th>
      <th>wikidata</th>
      <th>wikidata_group</th>
      <th>wikidata_area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Amy Klobuchar</td>
      <td>Klobuchar, Amy</td>
      <td>NaN</td>
      <td>SenAmyKlobuchar</td>
      <td>NaN</td>
      <td>Democrat</td>
      <td>democrat</td>
      <td>ocd-division/country:us/state:mn</td>
      <td>Minnesota</td>
      <td>Senate</td>
      <td>116</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://theunitedstates.io/images/congress/ori...</td>
      <td>female</td>
      <td>Q22237</td>
      <td>Q29552</td>
      <td>Q1527</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Angus S. King, Jr.</td>
      <td>King, Angus</td>
      <td>NaN</td>
      <td>SenAngusKing</td>
      <td>SenatorAngusSKingJr</td>
      <td>Independent</td>
      <td>independent</td>
      <td>ocd-division/country:us/state:me</td>
      <td>Maine</td>
      <td>Senate</td>
      <td>116</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://theunitedstates.io/images/congress/ori...</td>
      <td>male</td>
      <td>Q544464</td>
      <td>Q327591</td>
      <td>Q724</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ben Sasse</td>
      <td>Sasse, Benjamin</td>
      <td>NaN</td>
      <td>SenSasse</td>
      <td>SenatorSasse</td>
      <td>Republican</td>
      <td>republican</td>
      <td>ocd-division/country:us/state:ne</td>
      <td>Nebraska</td>
      <td>Senate</td>
      <td>116</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://theunitedstates.io/images/congress/ori...</td>
      <td>male</td>
      <td>Q16192221</td>
      <td>Q29468</td>
      <td>Q1553</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Benjamin L. Cardin</td>
      <td>Cardin, Benjamin</td>
      <td>NaN</td>
      <td>SenatorCardin</td>
      <td>senatorbencardin</td>
      <td>Democrat</td>
      <td>democrat</td>
      <td>ocd-division/country:us/state:md</td>
      <td>Maryland</td>
      <td>Senate</td>
      <td>116</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://theunitedstates.io/images/congress/ori...</td>
      <td>male</td>
      <td>Q723295</td>
      <td>Q29552</td>
      <td>Q1391</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bernard Sanders</td>
      <td>Sanders, Bernard</td>
      <td>NaN</td>
      <td>SenSanders</td>
      <td>senatorsanders</td>
      <td>Independent</td>
      <td>independent</td>
      <td>ocd-division/country:us/state:vt</td>
      <td>Vermont</td>
      <td>Senate</td>
      <td>116</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://theunitedstates.io/images/congress/ori...</td>
      <td>male</td>
      <td>Q359442</td>
      <td>Q327591</td>
      <td>Q16551</td>
    </tr>
  </tbody>
</table>
</div>



One can run this code in order to see the dataset used for this project.

5. The user can perform a regression analysis with the sentiment scores using the scores as the dependent variable and using political affilation and gender as the independent variable. If the user has binary variables of interest, one could create dummy variables. In order to create dummy variables, change the column of interest`group` and `gender` to include your own variables. These variables can be coded into 1's and 0's in order to perform regression analysis.

```python
senators.loc[senators["group"] == "Democrat", 'Democrat'] = 1
senators.loc[senators["group"] == "Republican", 'Democrat'] = 0
senators.loc[senators["gender"] == "female", 'female'] = 1
senators.loc[senators["gender"] != "female", 'female'] = 0

```

6. After cleaning the dataset, [regression](https://github.com/joshuajacobs2020/QTM-350---Group-4/blob/main/Regression%20Results.ipynb)  can be used in order to analyze the relatonship between political affilation/gender and sentiment scores. A regression was done for positive, negative, negative and neutral phrases.

![image.png](Regression Positive Score.png](https://github.com/joshuajacobs2020/QTM-350---Group-4/blob/4459f4a355ad7c954135a7a80aedffe2739a190b/Regression%20Positive%20Score.png)

Here is an example of regression results for positive phrases.

7. A better way to analyze the results are through[visualizations](https://github.com/joshuajacobs2020/QTM-350---Group-4/blob/main/Data%20Visualizations.ipynb). In this file, we created box and whisker plots to hone in on the median positive and negative score for different political affiliations. An example of creating a boxplot is provided. If the user is creating their own data visualizations, the x parameter can be used to plot their own data generated from the sentiment score.

```python
ax.boxplot(x=[Dem3['Negative'], Rep3['Negative']], vert=False)
ax.set_yticklabels(['Democrats', 'Republicans'])
ax.set_title('Political Party Negative Scores')
ax.text(0.04, 0.2, Dem_median_3, transform=ax.transAxes, fontsize=7, va='top')
ax.text(0.069, 0.7, Rep_median_3, transform=ax.transAxes, fontsize=7, va='top')
plt.show()
plt.savefig('Negative.png')
```

## File Description

The main files that will be useful to reproduce this work are:


[Blog.md](https://github.com/joshuajacobs2020/QTM-350---Group-4/blob/main/Blog.md) - Summarizes the problem, background, methods, and analysis in our blog post

[ComprehendGuide.ipynb](https://github.com/joshuajacobs2020/QTM-350---Group-4/blob/main/ComprehendGuide.ipynb) - An example of a specfic use case to analyze the sentiment of an entity from a phrase

[Data Visualizations.ipynb](https://github.com/joshuajacobs2020/QTM-350---Group-4/blob/main/Data%20Visualizations.ipynb) - A visualization from the results of the regression which includes box and whisker plots of the political party's sentiment score 

[DataCreation.ipynb](https://github.com/joshuajacobs2020/QTM-350---Group-4/blob/main/DataCreation.ipynb) - Generating the dataset with a comprehensive overview of how to devise scores for entities. It includes creating dummy variables for variables that are binary, generating different phrases for each entity, and merging the results to be able to perform regression analysis

[Regression Results.ipynb](https://github.com/joshuajacobs2020/QTM-350---Group-4/blob/main/Regression%20Results.ipynb) - This file shows the results of the regression analysis which shows the ANOVA table for positive, negative and neutral scores. The variance has also been calculated to show the precision of the coefficients.

[Datasets](https://github.com/joshuajacobs2020/QTM-350---Group-4/tree/main/datasets) - The datasets include a file from [Every Politician](https://everypolitician.org/united-states-of-america/senate/download.html) and updated datasets created from python.




### AWS SDK for Python to use Amazon Comprehend


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

![Logo](https://media.amazonwebservices.com/blog/2017/comp_logo_2.png)

## Contact

Austin Cherian  - austin.zach.cherian@emory.edu 🚀\
William Coupe - joshua.jacobs@emory.edu  😀\
Josh Jacobs - joshua.jacobs@emory.edu 👍\
Eugene Lim - elim27@emory.edu 💯\
Ved Udare - ved.ravindra.udare@emory.edu 🤗

Project Link: https://github.com/joshuajacobs2020/QTM-350---Group-4

