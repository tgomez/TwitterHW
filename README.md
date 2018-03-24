

```python
# 3 Observations
# 1- The last 100 tweets for all five news outlets analyzed have an overall negative sentiment.
# 2- BBCWorld has the most negative score of the five new outlets analyzed.
# 3- Fox News has the least negative score of the five new outlets analyzed.
```


```python
#Import Dependencies, keys, etc
import tweepy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

from APITweeter2 import consumer_key
from APITweeter2 import consumer_secret
from APITweeter2 import access_token
from APITweeter2 import access_token_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
#Getting tweets and runing vader analysis
target_users = ["@BBCWorld", "@CBSNews", "@CNN", "@FoxNews","@nytimes"]

outlets = []
compound_list = []
positive_list = []
negative_list = []
neutral_list = []
tweet_counter = []

for target in target_users:

        public_tweets = api.user_timeline(target, count=100, result_type="recent")
        
        tweet_number = 100

        
        for tweet in public_tweets: 

            scores = analyzer.polarity_scores(tweet['text'])
            compound = scores['compound']
            pos = scores['pos']
            neu = scores['neu']
            neg = scores['neg']
            
            tweet_number -= 1
                  
            compound_list.append(compound)
            positive_list.append(pos)
            negative_list.append(neg)
            neutral_list.append(neu)
            outlets.append(target)
            tweet_counter.append(tweet_number)
            
            sentiments = {"User": outlets,
                          "Date": tweet["created_at"],
                          "Compound": compound_list,
                          "Positive": positive_list,
                          "Negative": neutral_list,
                          "Neutral": negative_list,
                         "Tweet_Count": tweet_counter}
       
```


```python
sentiments_pd = pd.DataFrame.from_dict(sentiments)
sentiments_pd.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweet_Count</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>Fri Mar 23 10:30:10 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>99</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.4019</td>
      <td>Fri Mar 23 10:30:10 +0000 2018</td>
      <td>0.722</td>
      <td>0.278</td>
      <td>0.0</td>
      <td>98</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.3818</td>
      <td>Fri Mar 23 10:30:10 +0000 2018</td>
      <td>0.658</td>
      <td>0.342</td>
      <td>0.0</td>
      <td>97</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>Fri Mar 23 10:30:10 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>96</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>Fri Mar 23 10:30:10 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>95</td>
      <td>@BBCWorld</td>
    </tr>
  </tbody>
</table>
</div>




```python
sentiments_pd.to_csv('TwitterHW.csv')
```


```python
user_sentiments = sentiments_pd.pivot(index="Tweet_Count", columns="User", values="Compound")
user_sentiments.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>User</th>
      <th>@BBCWorld</th>
      <th>@CBSNews</th>
      <th>@CNN</th>
      <th>@FoxNews</th>
      <th>@nytimes</th>
    </tr>
    <tr>
      <th>Tweet_Count</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.3400</td>
      <td>0.3400</td>
      <td>0.0000</td>
      <td>0.3291</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0000</td>
      <td>-0.5994</td>
      <td>-0.6590</td>
      <td>0.0772</td>
      <td>-0.5267</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.5859</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0772</td>
      <td>-0.0772</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.6249</td>
      <td>-0.8720</td>
      <td>0.0000</td>
      <td>0.3612</td>
      <td>-0.5574</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>0.3400</td>
      <td>-0.6705</td>
      <td>0.2263</td>
      <td>0.6369</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Scatter Plot

x_values = np.arange(100)

fig = plt.figure(figsize=(10, 5))

for user in target_users:
    
    plt.scatter(x_values, user_sentiments[user], marker="o", alpha=0.9)


plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
now = datetime.datetime.now()
month = now.strftime("%b")
plt.title(f"VADER Sentiment Analysis of Tweets {now.day} {month} {now.year}")
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
plt.grid(True)

plt.savefig('TwitterHWScatter.png')

plt.show()
```


![png](output_6_0.png)



```python
#Arrange data for bar graphs
grouped_news = sentiments_pd.groupby('User')

grouped_compound = grouped_news['Compound'].mean()

```


```python
# Create Bar Graph
x_values = np.arange(len(target_users))

plot_data = zip(x_values, target_users)

fig = plt.figure(figsize=(10, 5))

for x, user in plot_data:
    
    y = grouped_compound[user]
    
    plt.bar(x, y)
    plt.text(x, y/2, '{:.5}'.format(y),
             horizontalalignment='center', color='black',
             fontsize=13, weight='bold')

plt.xticks(x_values, target_users)
now = datetime.datetime.now()
month = now.strftime("%b")
plt.title(f"Overall Media Sentiment based on Twitter {now.day} {month} {now.year}")
plt.ylabel("Tweet Polarity")
plt.xlabel("Twitter Account")
plt.savefig('TwitterHWBar.png')
plt.show()
```


![png](output_8_0.png)

