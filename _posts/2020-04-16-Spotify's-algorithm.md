---
layout:     post
title:      Spotify's Algorithm
date:       2020-04-16 08:19:29
summary:    Introduction to Spotify's recommendation system.
categories: python, deep learning
---

## Spotify's Algorithm

What’s the biggest name in the music streaming industry today? If you guessed Spotify, you guessed it right. It has over 100 million paid subscribers! The key highlight about Spotify is the way it is personalised for each user. The more you use Spotify, the better it knows you and the better it recommends!

![ss11.png](https://djinit-ai.github.io/images/ss11.png)

It has a “Discover Weekly” playlist, which is an absolute gem for music lovers. It curates a mixtape of 30 absolutely fresh songs for you, totally based on your streaming history. These songs are ones that you’ve never listened to, but will totally strike a chord with you. We’re going to see how Artificial Intelligence makes this magic happen.

Spotify’s recommendation system is built on the belief to provide each and every user a personalised feel and a sense of uniqueness to his/her listening experience. Its goal is to quickly help users find something they are going to enjoy listening to, according to a presentation by Spotify Research director Mounia Lalmas-Roelleke at the Web Conference earlier this year.

![ss22.png](https://djinit-ai.github.io/images/ss22.png)

A Spotify user’s home screen is governed by an A.I. system called BaRT (“Bandits for Recommendations as Treatments”). BaRT is solely responsible for organising a users complete display  and collections included shelves of suggested songs that follow a theme like “best of artists” or “keep the vibe going,” and the order the playlists appear on those shelves that a user might like. But at the same time, BaRT is responsible for providing the user with new and fresh content regularly.

Spotify adopts the ‘Exploit & Explore’ mechanism for providing user recommendations. BaRT “exploits” a given users data to analyse various facts like his/her 
- music listening history
- which songs you’ve skipped
- what playlists you’ve made
- your activity on the platform’s social features
- and EVEN YOUR LOCATION !

After using all of this data, BaRT starts understanding trends in users listening behaviour, which for humans would mean ‘understanding one’s taste’. BaRT learns the users likes and dislikes, and depending on that, it decides what else might that user like hear.

In order to create Discover Weekly, there are three main types of recommendation models that Spotify employs:
- Collaborative Filtering models such as the ones that originally used. It will analyze both your behavior and others’ behaviors based on all the above mentioned features.
- Natural Language Processing (NLP) models which work to analyze text to provide you songs with a similar background and description.
- Audio models which related to analyze the raw audio tracks themselves.

**Collaborative Filtering:**
Collaborative Filtering is a technique where Spotify provides you with recommendations not just depending on what you hear, but also what people all over Spotify hear. Depending on a few songs that a user might like, Spotify tries to analyse his/her behaviour. Depending on this analysis, it tries to find out -- ‘What kind of songs did those people hear who also liked the songs of our user under analysis. Let’s take an example.

Consider the whole Spotify community just confined to 4 people, Sheldon, Leonard, RAj and Howard. All of them have their own taste in music. For simplicity, let’s assume that the Spotify Database is just confined to five songs. To make things simpler, the only dimensions we will be considering is how many times a user listens to a particular song. That’s good enough to begin with!

![ss33.png](https://djinit-ai.github.io/images/ss33.png)

This is all the data we need! As shown above, each user has heard these 5 songs a given number of times and they all clearly have their likes and dislikes. However, a few anomalous users can cause a skew in our analysis and so scaling our data becomes important.

![ss44.png](https://djinit-ai.github.io/images/ss44.png)

That looks better! Now for every user, the data about how many times he listens to each song is normalized around it’s mean.
Next up, we will be working with a value which actually helps a machine understand how similar two things are. As the name suggests, this value is called the Cosine Similarity.

**Cosine Similarity:**

The best way to understand Cosine Similarity is with the help of an example. Let’s say you have 3 songs A, B and C. Each song has only 2 dimensions, namely hits and ranking.Given to us is the following information, A has 20 hits and a ranking of 5, B has 50 hits and a ranking of 4 and C has 80 hits and a ranking of 2.

We know that the “cos” of an angle between two vectors is 0 when the vectors face opposite to each other and 1 when they coincide. This fact leads us to the thought that the similarity between 2 vectors and the cos of the angle between them is directly proportional. Hence the name COSINE - SIMILARITY !!

![ss55.png](https://djinit-ai.github.io/images/ss55.png)

This is the reason we can say that song B and song A are really similar both graphically and by considering the cos of their angles between them.

Now imagine such a situation with way more dimensions than hits and ranking and surely a million more songs! This is how any recommendation engine understands the similarity between 2 comparable entities.

Getting back to our example, we have understood how a machine understands the level of similarity between 2 songs...But how does it really use it?

![ss66.png](https://djinit-ai.github.io/images/ss66.png)

What the piece of code above essentially does is that it treats the no. of times each individual has heard a particular songs as dimensions of each song and on the basis of these dimensions try to find how similar one song is to the other.

Confused? Don’t worry, just take one song at a time, say Havana - for this songs now, the dimensions are {#(Sheldon heard it), #(Leonard heard it), #(Howard heard it), #(Raj heard it)}. This is similar to the dimensions we had in the earlier example, ie. Hits and Rating. This is done for all the other song and as a result the above heatmap is created.

Congratulations!! We built a recommendation system. Now all that’s left to do is suggest songs to new users. Let’s say our new user (obviously Penny) has already heard “Wake Me Up” 3 times. From this data, we can easily run a function to suggest the kind of songs she should hear next.

![ss77.png](https://djinit-ai.github.io/images/ss77.png)

And there you have it! Our recommendation system is up and suggesting like a pro. This essentially what Spotify, or any other platform where recommendations are required, uses. The only difference is that it uses data from a lot more users, with many more features extracted from each user (not just the no. of times he hears a song) like chosen languages, region, etc. This created a multi-dimensional grid which uses the same principle as above to provide better decisions.

Meanwhile the main ingredient in Discover Weekly is the people who listens itself. Spotify begins by looking at the 2 billion or so playlists created by its users and each one be as a reflection of some music fan’s tastes and sensibilities. Those human selections and groupings of songs form the core of Discover Weekly’s recommendations.

But when Spotify explores, it uses the information about the rest of the world, like playlists and artists similar to your taste in music but you haven’t heard yet, the popularity of other artists, and more. In the exploring process, BaRT is responsible for providing users with content which is relative to him/her, but at the same time expose a user to the new and trending material out there ( Because that’s what brings in the money at the end of the day ;) ).

![ss88.png](https://djinit-ai.github.io/images/s88.png)

Success for BaRT is measured by whether you actually listen to the suggested music and for how long. When you stream a song for more than 30 seconds, the algorithm tracks that as getting the recommendation right, according to the presentation. The longer you listen to the recommended playlist or set of songs, the better the recommendation is determined to be.Spotify’s sweet spot for understanding whether a person likes a song or not seems to be 30 seconds. In a 2015 interview with Quartz, Spotify product director Matthew Ogle, who has since left the company, mentioned that skipping before the 30-second mark is the equivalent of a thumbs down for the Discover Weekly playlist.



**How does Spotify collects user’s data?**

Whatever we do on spotify , they transform it into useful data which can be used by them to provide better recommendations to the user. 

**Different ways in which they collect are data are listed below:**
- Whenever we sign in to Spotify Services they collect certain personal data so that we can use the Spotify Service such as ouu email address , phone number, birth date, gender, and country etc.
- Whenever we use the Spotify Service, they collect personal data about our use of the Spotify Service, such as what songs we have played and what playlists you have created.Now this data collection is not only beneficial for making predictions for you but this data also helps them to provide recommendations to different users who share the same taste in  music and providing a betterment to each and every user.
- From time to time, they may ask us to provide them with additional personal data or give them our permission to collect additional personal data e.g. to provide you with more features or functionality but this additional data does not include collection of our  photos, precise mobile device location, voice data, or contacts from our device without your prior consent as mentioned in their privacy policy. But however at anytime if we feel insecure of out data of getting expose we are frento turn of this future anytime whenever required.
- This is the most important source for them to collect data in which they collect user’s data from a third party source.They receive personal data about us and our activity from third parties, including advertisers and partners we work with in order to provide us with the Spotify Service. They use this personal data either where we have provided our consent to the third party or to Spotify to that data sharing taking place or where Spotify has a legitimate interest to use the personal data in order to provide us with the Spotify Service.


Example:- Suppose we are browsing through our news feed on facebook and there may be a song posted by many of your friends and you may have given your like for that song , then Spotify make sures that song and other songs related to the earlier one is shown at the top whenever u log into your spotify account.


![ss99.png](https://djinit-ai.github.io/images/ss99.png)

![ss10.png](https://djinit-ai.github.io/images/ss10.png)

Now you know how Artificial Intelligence is implemented to get you killer recommendations on your Spotify. And that is what has given Spotify this huge user base it has today!
