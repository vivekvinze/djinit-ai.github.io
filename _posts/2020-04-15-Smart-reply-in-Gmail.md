---
layout:     post
title:      Getting started with TensorFlow.js
date:       2020-02-11 12:19:29
summary:    Introduction to TensorFlow.js
categories: python,tensorflow, JavaScript
---

## Smart Reply in Gmail

It’s pretty easy to read your emails while you’re on the go, but responding to those emails takes effort. Smart Reply, 
​available in Inbox by Gmail , saves you time by suggesting quick responses to your messages. The feature drives 12 percent 
of replies in Inbox on mobile. And it has also been started for Android and iOS .

![Image of Yaktocat](https://miro.medium.com/max/848/1*kgzLawJmfp3i3UCG_KhfDA.png)


From the above we can see that the system can find responses that are on point, without an overlap of keywords or even 
synonyms of keywords.It’s delighting to see the system suggesting results that show understanding and are helpful.


## Smarts behind Smart Reply
**How does it suggest brief responses when appropriate, just one tap away?**
It uses the the sequence-to sequence learning framework , which uses long short term memory networks (LSTMs) to predict sequences of text. Consistent with the approach of the Neural Conversation Model,the input sequence is the content of the email and output being all possible replies.Smart Reply consists of the following components, which are also shown in Fig1
