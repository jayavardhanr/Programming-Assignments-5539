from __future__ import print_function

try:
    import json
except ImportError:
    import simplejson as json
    
from collections import Counter
import operator
import numpy as np
import matplotlib.pyplot as plt
import pprint
import langid

tweets_considered=15000

print('Loading file twitter_stream_15000tweets.txt, which has worldwide tweets.')
print('Note: Refer twitter_streaming.py for the code')
print('\n')

#Counting the deleted and proper tweets in 15000 tweets stored in file
tweets_filename = 'twitter_stream_15000tweets.txt'
tweets_file = open(tweets_filename, "r")
tweet_count = tweets_considered
delete_count,geo_enabled_count,proper_tweets,tweets_with_langID=0,0,0,0
languages_twitter_api=[]
languages_langid=[]

#langid predictions
langid_mismatch=[]
langid_match=[]

for line in tweets_file:
    tweet_count -= 1    
    try:
        # Read in one line of the file, convert it into a json object 
        tweet = json.loads(line.strip())
        if 'delete' in tweet:
            delete_count+=1
        
        elif 'text' in tweet:
            if 'lang' in tweet.keys():
                tweets_with_langID+=1
                languages_twitter_api.append(tweet['lang'])
                
            #langid
            langid_prediction=langid.classify(tweet['text'])[0]
            languages_langid.append(langid_prediction)
            if langid_prediction!=tweet['lang']:
                langid_mismatch.append((tweet['text'],tweet['lang'],langid_prediction))
            else:
                langid_match.append((tweet['text'],tweet['lang'],langid_prediction))
            
            ##Geotagging
            if tweet['user']['geo_enabled']:
                geo_enabled_count+=1
                
            proper_tweets+=1
        
        else:
            print('Other kind of tweets from API',tweet.keys())

    except Exception as e:
        # read in a line is not in JSON format (sometimes error occured)
        print("error",e,line)
        continue
        
    finally:
        if tweet_count <= 0:
            break 
            

print('Information:')
print('delete tweets: tweets which are tagged delete by Twitter API')
print('limit tweet tags: Twitter streaming gives some limit tags along with tweets while filtering based on coordinates')
print('proper tweets: tweets excluding the (delete+limit) tweets, which contain proper Text field')

print('\n')
print('Total number of Tweets:',tweets_considered)
#print('Total number of delete tweets:',delete_count)
print('Total number of proper tweets :',proper_tweets)
#print("percentage of tweets deleted:",float(delete_count*100.0/tweets_considered))
#print("percentage of proper tweets:",float(proper_tweets*100.0/tweets_considered))

#print('Total number of tweets with langId:',tweets_with_langID)
print('\n')
print('Question 2(a)')
print("Percentage of tweets with langId (in total tweets):",float(tweets_with_langID*100.0/tweets_considered))
print("Percentage of tweets with langId(in proper tweets):",float(tweets_with_langID*100.0/proper_tweets))

##finding the percentage of each language tag provided by Twitter API
print('\n')
language_count_twitter_api=Counter(languages_twitter_api)
language_percentage_twitter_api={}
s = sum(language_count_twitter_api.values())
for k, v in language_count_twitter_api.items():
    pct = v * 100.0 / s
    language_percentage_twitter_api[k]=pct

print('\n')
print('Question 2(b)')
print('Total number of languages tags provided by twitter API:',len(language_percentage_twitter_api.keys()))
print('\n')
print('Question 2(c)')
print('Percentage of each language tag(by twitter API):')
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(language_percentage_twitter_api)


##finding the percentage of each language tag provided by LangId API
print('\n')
language_count_langid=Counter(languages_langid)
language_percentage_langid={}
s = sum(language_count_langid.values())
for k, v in language_count_langid.items():
    pct = v * 100.0 / s
    language_percentage_langid[k]=pct

print('Question 3(a)')
print('Total number of languages tags provided by Langid API:',len(language_percentage_langid.keys()))


##Langid vs Twitter API
print('\n')
#print('Twitter API vs Langid API')
print('Question 3(b)')
print('Total number of mismatches between Langid and twitter API:',len(langid_mismatch))
print('Total number of matches between Langid and twitter API:',len(langid_match))
print('percentage of langid vs twitter API mis-matches( in proper tweets):',float(len(langid_mismatch)*100.0/proper_tweets))
print('percentage of langid vs twitter API matches( in proper tweets):',float(len(langid_match)*100.0/proper_tweets))


#print('percentage of each language tag(by Langid API):')
#pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(language_percentage_langid)
print('\n')
print('Question 3(c)')
print('The twitter API and langid.py disagree for the following cases:')
print('1) tweets with very little text')
print('2) pure hyperlink tweets and also tweets where hyperlink dominates the tweet content')
print('3) Tweets which have a lot of unicode characters')
#print('\n')
print('Additional Points:')
print('1) Twitter api performs better on english tweets with hyperlinks compared to langid API')
print('2) Both of them agree on tweets with good text content')
print('\n')

#Tweets in USA(using Steaming API) 
print('Loading twitter_stream_US_15000tweets.txt, which has tweets filtered out for USA')
print('please refer: twitter_streaming_USA.py for the code')
print('\n')

tweets_filename = 'twitter_stream_US_15000tweets.txt'
tweets_file = open(tweets_filename, "r")
tweets_considered=15000
tweet_count = tweets_considered
proper_tweets_usa,tweets_with_langID,geo_enabled_count_usa=0,0,0
languages=[]

for line in tweets_file:
    tweet_count -= 1    
    try:
        # Read in one line of the file, convert it into a json object 
        tweet = json.loads(line.strip())
        if 'delete' in tweet:
            pass
        
        elif 'text' in tweet:
            if 'lang' in tweet.keys():
                tweets_with_langID+=1
                languages.append(tweet['lang'])
            
            if tweet['user']['geo_enabled']:
                geo_enabled_count_usa+=1
                
            proper_tweets_usa+=1
        
        else:
            #print(json.dumps(tweet,indent=4,sort_keys=True))
            #print('other variety tweets from streaming Api',tweet.keys())
            pass
    except Exception as e:
        # read in a line is not in JSON format (sometimes error occured)
        print("error",e,line)
        continue
    finally:
        if tweet_count <= 0:
            break 

#print('\n')
#print('Information')
#print('Twitter streaming gives some limit tags along with tweets while filtering based on coordinates')
print('\n')
print('Total number of Tweets:',tweets_considered)
print('Total number of proper tweets :',proper_tweets_usa)
#print("percentage of proper tweets:",float(proper_tweets_usa*100.0/tweets_considered))
#print('Total number of tweets with langId:',tweets_with_langID)
#print("Percentage of tweets with langId (in limit + proper tweets):",float(tweets_with_langID*100.0/tweets_considered))
#print("percentage of tweets with langId(in proper tweets):",float(tweets_with_langID*100.0/proper_tweets_usa))

##finding the percentage of each language tag provided by twitter API for tweets in USA
print('\n')
language_count=Counter(languages)
language_count_per={}
s = sum(language_count.values())
for k, v in language_count.items():
    pct = v * 100.0 / s
    language_count_per[k]=pct
#print('Total number of languages tags provided by twitter API for tweets in USA:',len(language_count_per.keys()))
print('\n')
print('Question 4(a)')
print('percentage of each language tag for tweets in USA(by twitter API):')
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(language_count_per)

##GeoTag
## Percentage of tweets which are geotagged
print('\n')
print('Question 4(b)')
print('Total number of tweets which are geotagged:',geo_enabled_count)
print("Percentage of tweets with geotagged:",float(geo_enabled_count*100.0/tweets_considered))
print("Percentage of tweets with geotagged(in proper tweets):",float(geo_enabled_count*100.0/proper_tweets))

## Percentage of tweets which are geotagged in USA
print('\n')
print(' Total number of tweets which are geotagged in USA:',geo_enabled_count_usa)
print(" percentage of tweets with geotag in USA:",float(geo_enabled_count_usa*100.0/tweets_considered))
print(" percentage of tweets with geotag in USA(in proper tweets):",float(geo_enabled_count_usa*100.0/proper_tweets_usa))

print('\n')
print('Note: All proper tweets in USA are geotagged, as we use coordinates information to retrieve tweets for USA. The \
coordinates field is populated only when the tweets are geotagged')

#Plots

#Comparision Plots between Twitter API vs Langid.py and worldwide vs USA language tags
language_percentage_twitter_api_sorted=sorted(language_percentage_twitter_api.items(), key=operator.itemgetter(1),reverse=True)
langid_percentages=[]
twitter_api_percentages=[]
tweets_in_usa_percentage=[]
languages_detected_by_twitter=[]
for language,percentage in language_percentage_twitter_api_sorted[:20]:
    languages_detected_by_twitter.append(language)
    twitter_api_percentages.append(percentage)
    if language in language_percentage_langid.keys():
        langid_percentages.append(language_percentage_langid[language])
    else:
        langid_percentages.append(0)
        
    if language in language_count_per.keys():
        tweets_in_usa_percentage.append(language_count_per[language])
    else:
        tweets_in_usa_percentage.append(0)  

#Comparision Plots between Twitter API vs Langid.py
ax = plt.subplot(111)
w = 0.4
pos1 = np.arange(len(languages_detected_by_twitter))
rects1 = ax.bar(pos1-w, twitter_api_percentages, width=w, color='b',align='center')
rects2 = ax.bar(pos1, langid_percentages, width=w, color='g',align='center')
ax.legend(('twitter API', 'LangID.py'),fontsize=14)
ax.set_xticks(pos1-w/2)
plt.xlabel('Language tags detected for tweets', fontsize=14)
plt.ylabel('Percentage of each language(%)', fontsize=14)
ax.set_xticklabels(languages_detected_by_twitter,fontsize=12)
plt.title('Comparision of language tags, Twitter API vs langid.py',fontsize=20)
ax.autoscale(tight=True)
plt.show()

#Comparision Plots of language Tags, worldwide vs USA
ax = plt.subplot(111)
w = 0.4
pos1 = np.arange(len(languages_detected_by_twitter))
rects1 = ax.bar(pos1-w, twitter_api_percentages, width=w, color='b',align='center')
rects2 = ax.bar(pos1, tweets_in_usa_percentage, width=w, color='g',align='center')
ax.legend(('Worldwide', 'USA'),fontsize=14)
ax.set_xticks(pos1-w/2)
plt.xlabel('Language tags detected for tweets', fontsize=14)
plt.ylabel('Percentage of each language(%)', fontsize=14)
ax.set_xticklabels(languages_detected_by_twitter,fontsize=12)
plt.title('Comparision of language tags, worldwide vs USA',fontsize=20)
ax.autoscale(tight=True)
plt.show()

#Old plots

##Plotting Figure for percentage of Languages by Twitter API
print('\n')
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 20
fig_size[1] = 10
plt.xlabel('Language Tags detected by Twitter API', fontsize=17)
plt.ylabel('Percentage of each Language(%)', fontsize=17)
plt.rcParams["figure.figsize"] = fig_size
pos1 = np.arange(len(language_percentage_twitter_api.keys()))
width = 1.0 
ax = plt.axes()
ax.set_xticks(pos1)
ax.set_xticklabels(language_percentage_twitter_api.keys(),fontsize=13)
plt.bar(pos1, language_percentage_twitter_api.values(), width, color='r')
plt.show()


##Plotting Figure for percentage of Languages by LangId API
print('\n')
plt.xlabel('Language Tags detected by Langid API', fontsize=17)
plt.ylabel('Percentage of each Language(%)', fontsize=17)
plt.rcParams["figure.figsize"] = fig_size
pos1 = np.arange(len(language_percentage_langid.keys()))
width = 1.0
ax = plt.axes()
ax.set_xticks(pos1)
ax.set_xticklabels(language_percentage_langid.keys(),fontsize=13)
plt.bar(pos1, language_percentage_langid.values(), width, color='r')
plt.show()

##Plotting Figure for percentage of Languages by twitter API for tweets in USA
print('\n')
plt.xlabel('Language Tags detected for tweets in USA', fontsize=17)
plt.ylabel('Percentage of each Language(%)', fontsize=17)
plt.rcParams["figure.figsize"] = fig_size
pos1 = np.arange(len(language_count_per.keys()))
width = 1.0
ax = plt.axes()
ax.set_xticks(pos1)
ax.set_xticklabels(language_count_per.keys(),fontsize=13)
plt.bar(pos1, language_count_per.values(), width, color='r')
plt.show()