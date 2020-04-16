---
layout:     post
title:      Spotify's Algorithm
date:       2020-04-16 08:19:29
summary:    Introduction to smart reply system in Gmail
categories: python
---

## Spotify's Algorithm

What’s the biggest name in the music streaming industry today? If you guessed Spotify, you guessed it right. It has over 100 million paid subscribers! The key highlight about Spotify is the way it is personalised for each user. The more you use Spotify, the better it knows you and the better it recommends!

![Image of Yaktocat](https://github.com/djinit-ai/djinit-ai.github.io/blob/master/images/ss1.png)

It has a “Discover Weekly” playlist, which is an absolute gem for music lovers. It curates a mixtape of 30 absolutely fresh songs for you, totally based on your streaming history. These songs are ones that you’ve never listened to, but will totally strike a chord with you. We’re going to see how Artificial Intelligence makes this magic happen.

Spotify’s recommendation system is built on the belief to provide each and every user a personalised feel and a sense of uniqueness to his/her listening experience. Its goal is to quickly help users find something they are going to enjoy listening to, according to a presentation by Spotify Research director Mounia Lalmas-Roelleke at the Web Conference earlier this year.

![Image of Yaktocat](https://github.com/djinit-ai/djinit-ai.github.io/blob/master/images/ss1.png)

A Spotify user’s home screen is governed by an A.I. system called BaRT (“Bandits for Recommendations as Treatments”). BaRT is solely responsible for organising a users complete display  and collections included shelves of suggested songs that follow a theme like “best of artists” or “keep the vibe going,” and the order the playlists appear on those shelves that a user might like. But at the same time, BaRT is responsible for providing the user with new and fresh content regularly.

Spotify adopts the ‘Exploit & Explore’ mechanism for providing user recommendations. BaRT “exploits” a given users data to analyse various facts like his/her 
- music listening history
- which songs you’ve skipped
- what playlists you’ve made
- your activity on the platform’s social features
- and EVEN YOUR LOCATION !
After using all of this data, BaRT starts understanding trends in users listening behaviour, which for humans would mean ‘understanding one’s taste’. BaRT learns the users likes and dislikes, and depending on that, it decides what else might that user like hear.

In order to create Discover Weekly, there are three main types of recommendation models that Spotify employs:
Collaborative Filtering models such as the ones that originally used. It will analyze both your behavior and others’ behaviors based on all the above mentioned features.
Natural Language Processing (NLP) models which work to analyze text to provide you songs with a similar background and description.
Audio models which related to analyze the raw audio tracks themselves.
Collaborative Filtering:
Collaborative Filtering is a technique where Spotify provides you with recommendations not just depending on what you hear, but also what people all over Spotify hear. Depending on a few songs that a user might like, Spotify tries to analyse his/her behaviour. Depending on this analysis, it tries to find out -- ‘What kind of songs did those people hear who also liked the songs of our user under analysis. Let’s take an example.
Consider the whole Spotify community just confined to 4 people, Sheldon, Leonard, RAj and Howard. All of them have their own taste in music. For simplicity, let’s assume that the Spotify Database is just confined to five songs. To make things simpler, the only dimensions we will be considering is how many times a user listens to a particular song. That’s good enough to begin with!
