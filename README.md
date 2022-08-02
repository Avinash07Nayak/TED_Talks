TED Talks - EDA and Data Visualization
Humans are visual animals. We tend to understand visuals more clearly as compared to texts or sounds. This is true for Data as well. While analyzing Data, data points in some kind of visual format is more understandable than in tabular format or simple text format. To show data in visual format, we have many tools such as different kind of plots/graphs which help convert data in tabular format to visual format.![Tabular format of data](https://user-images.githubusercontent.com/110491966/182465520-79921a9c-c2d0-40fa-afa6-061d38f88cac.png)![Visual format of data](https://user-images.githubusercontent.com/110491966/182465587-ef538dd0-a971-4537-9539-d1086f790182.png)
Data in Tabular Format (Right) and in Visual Format (Left) [Credits: Google Image]TED is a non-profit organization devoted to spreading ideas, usually in the form of short, powerful talks (18 minutes or less). TED began in 1984 as a conference where Technology, Entertainment and Design converged, and today covers almost all topics - from science to business to global issues - in more than 100 languages.
TED is now a global community, welcoming people from every discipline and culture who seek a deeper understanding of the world. TED believes passionately in the power of ideas to change attitudes, lives and, ultimately, the world.
As an EDA and Data Visualization example, we will discuss about TED Talks here. We will first clean the data, do some preprocessing on it and then perform EDA.
The Dataset
The dataset can be easily found here. It is open sourced and is created by the author for simple data exploration purpose.
The Libraries
To perform Data cleaning, Data preprocessing and EDA, we will be using python libraries such as numpy, pandas, matplotlib etc. These libraries can be easily imported in the python environment using import keyword.
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import matplotlib.colors as cm
from wordcloud import WordCloud
from PIL import Image
Data Overview
The dataset is a single comma separated values (csv) file. It has 5,440 rows and 6 columns.
The column details are as follows:-
1. title: Title of the event/talk
2. author: Speaker of the event/talk
3. date: Date of the event/talk
4. views: Number of views the video of the event/talk got
5. likes: Number of likes the video of the event/talk got
6. link: URL link to the website having the video
Data Cleaning
Real world data has a lot of errors, missing values, abnormalities etc. We need to take care of them before proceeding ahead with preprocessing and EDA. Below are the steps we performed to clean the data for the given dataset
The dates before 1984 are most likely noise as TED was first started in 1984. We should remove the data-points whose event date is before 1984.

df.drop(list(df[pd.to_datetime(df['date']).dt.year<1984].index), axis=0, inplace=True)
2. Missing Data: The 'author' column has one missing data-point. 
df[df['author'].isnull()]
Since only one data-point has missing data, we can remove the entire row without effecting the dataset much.
df.dropna(axis=0, inplace=True)
3. Duplicate Data: We will check for duplicate data in the dataset. Now 'author', 'date', 'views' and 'likes' can have duplicate values. So we will check for duplicate data only for 'title' and 'link'. Also we will check for duplicate data of entire row.
df[df.duplicated()]
df[df.duplicated(subset=['link'])]
df[df.duplicated(subset=['title'])]
We didn't find any duplicate row or duplicate data in 'title' or 'link' column
Data Preprocessing
After cleaning the data, we need to preprocess it. Preprocessing involves doing some changes in the data so that its EDA can be performed with more ease and its model can provide with more accurate results. Below are the steps we performed to preprocess the cleaned data.
'link' column doesn't serve any meaningful purpose for analysis of the dataset. We can remove the column without much effect on the Dataset

df.drop(['link'], axis=1, inplace=True)
df.head()
2. FEATURE ENGINEERING: Weighted percentage of likes to views, weight being the Inverse of Number of Days passed from the Date of Posting of Video till 1st July 2022. A talk will have a high value of weighted percentage of likes to views if it is a recent talk and a high number of people who have viewed the talk have also liked it. A low value of this metric means that the talk is old or most people have viewed the talk but didn't liked it.
df['inv_days_passed']=pd.to_datetime('July 2022') - pd.to_datetime(df['date'], errors='coerce')
df['inv_days_passed']=df['inv_days_passed'].apply(lambda x: 1/(x.days))
df['likes/views (%)'] = (df['likes'].values/df['views'].values)*100
df['weighted_likes/views (%)'] = (df['inv_days_passed'].values)*(df['likes/views (%)'].values)*100
df.drop(['inv_days_passed'], axis=1, inplace=True)
df.head()
3. Segregating month and year from 'date' column and dropping the 'date' column
df['Month'] = pd.to_datetime(df['date']).dt.month_name()
df['Year'] = pd.to_datetime(df['date']).dt.year
df.drop(['date'], axis=1, inplace=True)
df.head()
4. 'title' column values may contain a lot of Stop-Words and special characters. We should get rid of them and store the preprocessed 'title' values in a new column. We then rearrange and rename the columns to get the final data-frame.
Exploratory Data Analysis
Year wise distribution of TED Talks
We plot a bar graph using Matplotlib library of python to get the Year wise distribution of TED Talks. This will show any relation between the number of TED talks conducted each year.

Year Wise distribution of TED TalksMost number of TED Talks happened in the year 2019
Number of TED Talks in the year 2020 and 2021 have decreased probably due to Covid-19
The number of TED talks have significantly increased in the year 2002 and in the year 2009 from it's previous years

2. Month wise distribution of TED Talks
We plot a bar graph to get the cumulative Month wise distribution of TED Talks. This will show any relation between the number of TED talks conducted every month.
Month Wise distribution of TED TalksMost number of TED Talks happened in the month of February and the least number of TED Talks happened in the month of January
The top 4 months are February, March, October and November

3. Top 10 authors with most number of TED Talks
We plot a bar graph to get the top 10 authors based on the number of TED Talks given by the authors.
Top 10 authors based on number of TED TalksAlex Gendler has the most number of TED Talks followed by Iseult Gillespie, Matt Walker and Alex Rosenthal

4. Top 10 authors with most views on TED Talks videos
We plot a bar graph to get the top 10 authors based on the cumulative number of views their TED Talks videos had got.
Top 10 authors based on cumulative number of views on their TED Talks videosAlex Gendler has the most Views on videos of TED Talks with cumulative views of 187,196,000
The difference between top author and second top author is huge (almost a jump of 100%)

5. Top 10 authors with most likes on TED Talks videos
We plot a bar graph to get the top 10 authors based on the cumulative number of likes their TED Talks videos had got.
Top 10 authors based on cumulative number of likes on their TED Talks videosAlex Gendler has the most Likes on videos of TED Talks with cumulative views of 5,691,000
The difference between top author and second top author is huge (almost a jump of 100%)

6. Top 10 most successful author on TED Talk
We plot a bar graph to get the top 10 most successful authors on TED talk. Success of an author is calculated by the cumulative sum of weighted percentage of Likes to Views of all the videos of the Author. Weighted percentage of a video is calculated as the Product of Ratio of Likes to Views and Inverse of Number of Days passed from the Date of Posting of Video till 1st July 2022 and multiplying it by 100.
Top 10 most successful authors on TED TalkThe most successful author is Iseult Gillespie followed closely by Alex Gendler, Matt Walker and Mona Chalabi

7. Count, Likes and Views of top 4 successful authors
We plot a cumulative bar graph to view the count (number of talks per author), likes (cumulative likes of the author on all videos) and views (cumulative views of the author on all videos) of the top 4 successful authors.
Counts (Number of talks), Likes and Views of top 4 most successful authorsFrom the above plot, we can find that by pure numbers Alex Gendler has lead in all the features but in terms of success given by cumulative sum of weighted ratio of likes to views (as we observed in previous plot), Iseult Gillespie takes the lead.

8. Top 10 most successful talks on TED Talks
We plot a bar graph to get the top 10 most successful talks on TED talk. Success of a talk is calculated by the cumulative sum of weighted percentage of Likes to Views of the video of the talk.
Top 10 most successful videos on TED TalkThe most successful talk is "How we're reducing the climate impact of electronics" by Tim Dunn followed closely by "How do jetpacks work? And why don't we all have them?" by Richard Browning, "The dark history of the overthrow of Hawaii" by Sydney Iaukea and "The myth of Narcissus and Echo" by Iseult Gillespie

9. Relationship between Likes and Views
Here we plot a scatter plot from Matplotlib to find any relation between the Likes and Views of the TED Talks.
Relationship between Likes and Views of TED TalksWe find that there is a very strong linear relationship between Views and Likes of the talks

10. Common Words in Title of the TED Talks
We plot a Word Cloud using wordcloud library in python. For this we first create a string variable with all the words of the title in it. We then pass the string variable to WordCloud().generate method to get the most common word in the form of a Word Cloud.
A Word Cloud showing the common word in title of TED TalksThe most common words used in the Title are WORLD, FUTURE, LIFE, MAKE, WORK, NEED etc.

Conclusion
EDA and Data Visualization is an important part in the life cycle of a Data Science project. It helps us understand and process the data as per our problem statement.
From above visualizations, we can say that:
Most of the TED Talks have happened in recent years with a considerable jump happening in the year 2009
Most of the talks took place in the month of February and November
Alex Gendler gave the most number of talks followed by Iseult Gillespie
Alex Gendler got the most number of total views and likes on the videos of his Talks followed by Sir Ken Robinson
Iseult Gillespie turned out to be the most Successful author followed closely by Alex Gendler and Matt Walker
"How we're reducing the climate impact of electronics" by Tim Dunn is the most Successful Talk followed closely by "How do jetpacks work? And why don't we all have them?" by Richard Browning
We found that Views and Likes have a very Strong Linear Relationship
Some of the most common Words used in Title are WORLD, FUTURE, LIFE, MAKE, WORK, NEED

Reference
Matplotlib - https://matplotlib.org
Pandas - https://pandas.pydata.org
TED - www.ted.com
Visualization help - https://www.kaggle.com/code/joshuaswords/netflix-data-visualization/
