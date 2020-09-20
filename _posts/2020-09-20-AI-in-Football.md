---
layout:     post
title:      Artificial Intelligence-The Future of Football
date:       2020-09-20 08:19:29
summary:    Implementations of AI in football
categories: python, computer vision
---

## Artificial Intelligence: The Future of Football

Football as a sport alone is beautiful. It is like a blank canvas waiting to be filled by the players on the field. It is a stunning spectacle, where we see elegance and athleticism, coupled with skills and intelligence, all for the simple goal of putting the ball in the net. This is why it’s referred to as <b>The Beautiful Game.</b><br>
Added to that, we see the beauty of time and relativity on display in the sport. 2 minutes become 10, and 10 minutes become 2, depending on the score. Finally the desire and urge to compete and win which the players display and the fans embody, makes for a great spectacle.
<br>
<p align="center">
<img src="/images/AI_football.png" height=500 width=750 />
</p>

For many centuries, the use of technology in football has been minimal, however seeing the advancements in the field of AI, one can already expect AI to be a very big part of Football.<br>
From using Computer Vision for getting tracking data to using plotting libraries and machine learning algorithms to make sense of the data, there is a lot of potential for such advancements.<br>
The current applications of AI that are being implemented in Football are Goal Line Technology (GLT), Video Assistant Referee (VAR), Electronic Performance and Tracking Systems(EPTS) and many more.<br>
<br>
### GOAL LINE TECHNOLOGY(GLT):
<br>
This Technology is used to check if the ball crosses the goal line or not. To implement this, multiple cameras are being used and the balls being used for the match are embedded with microchips. The cameras are used to track the movement of the ball and the output is then passed through heavy AI Algorithms which detect if the ball crossed the line or not.  It sends out a sound from the embedded microchip to the referee’s headset so they would know if the ball crossed the goal line or not. Advanced GLT can also pick up errors more quickly and accurately, which would otherwise have been missed by the human eye. 
<br>
<br>
<p align="center">
<img src="/images/GLT.png" height=400 width=600 />
</p>
<br>

### VIDEO ASSISSTANT REFEREE(VAR):
<br>
The current methods of using VAR to check offside is being heavily criticized and hence better approaches are being used with the most prominent one being the use of advanced AI algorithms.  Currently, optical tracking is commonplace in leagues around the world for performance analysis, and fans see it all the time through things like heatmaps. That optical tracking uses one point per player, but limb tracking follows between 15 and 20 points per player. Based on these data points, algorithms can work out which point, which limb, is closest to the goal line at any given moment, and can then use this to create the offside lines. This data, combined with artificial intelligence, is used to create a semi-automated offside system which detects the moment the ball is played and places the offside line in the correct position.<br>

### PLAYER PERFORMANCE ANALYSIS
At Loughborough University, a team of computer scientists led by Dr Baihua Li, have developed novel Artificial Intelligence Algorithms that are slowly but gradually changing the way team and individual player’s performance analysis on the pitch currently works.<br>
Current player performance analysis is a labour-intensive process and involves someone watching video recordings of matches and manually logging individual player’s actions which involves recording how many passes and shots were taken by a player, where the action took place, and whether it had a successful result.<br>
However the system developed by Dr Li and her team simplifies the human element behind tracking and showing insights and hence decreasing the error in accuracy.
The researchers have used the latest advances in computer vision, deep learning, and AI to achieve three main outcomes which are:<br>



### 1)Detecting body pose and limbs to identify actions:
<br>
<br>
<p align="center">
<img src="/images/LimbsDetect.png" height=400 width=600 />
</p>
<br>
Based on the current advancements in AI and Deep Learning, they have created an AI model wherein it keeps track of the movement of the players by detecting their body limbs and poses. The technology created watches the official video footage of the match, detects individual players or a group of players according to the requirement. State of the art Deep Learning and Computer Vision is being used to train the system.<br>
Deep learning involves getting a complex deep-layer neural network to learn hidden patterns and extract discriminative features from large amounts of data for perception.
In this case, the researchers used thousands of match recordings from all different football divisions -  that show various teams, poses,  jerseys, camera angles and background -  to train the AI to detect players and poses thus to recognise their movements, i.e. running, walking, kicking with their foot.<br>


### 2) Tracking players to get individual performance data
In addition to tracking the actions of the team, the neural network is also trained to track individual players and gather data throughout the match video.
<br>
<br>
<p align="center">
<img src="/images/camera.png" height=350 width=700 />
</p>
<br>
The system helps in tracking important data like how the entire team moves and whether they have cohesion, individual runs made by the players which helps them analyse their mistakes , movement on & off the ball and many more 

### 3) Camera stitching
Limited camera coverage (field of view) and low resolution have also been an issue when it comes to analysing lower league or grassroots games, as usually only low-cost affordable cameras are used to record a match. This is problematic as it is hard to record the whole field of view and the players can run in or out of the image view, so it is hard to track them.<br>
The researchers have come up with a solution to this; they propose using two low-cost consumable level normal cameras (such as GoPros), with each recording half of the football field, and a practical camera stitch method they have developed.<br>
The technology uses corresponding feature points from both cameras to generate a whole field of view – allowing players to be tracked and analysed much more reliably.<br>

### PLAYER AND FOOTBALL DETECTION USING OPENCV
Sadly, the above mentioned technology is not open source however a basic implementation of player and football detection can be implemented using OpenCV, masking and contour techniques.<br>

We use the France vs Belgium match clip as an example to perform player and football detection.<br>
We import the libraries, read the video and initialise the basic variables.<br>

<br>
<br>
<p align="center">
<img src="/images/code1.png" height=400 width=600 />
</p>
<br>

We now read the video frame by frame and convert them into HSV format.<br>
The HSV format helps us to separate the background by their colour. Hence we will be able to separate the pitch from the rest of the players and then detect the players. Also, after detecting the players we will be able to identify the national team they are playing by their jerseys.<br>

<br>
<br>
<p align="center">
<img src="/images/code2.png" height=400 width=600 />
</p>
<br>
After specifying the ranges we first define a mask of green colour to detect the pitch.<br>
<br>
<br>
<p align="center">
<img src="/images/code3.png" height=400 width=600 />
</p>
<br>

### OUTPUT:
<br>
<br>
<p align="center">
<img src="/images/code3_op.png" height=400 width=600 />
</p>
<br>
Now we will perform morphological closing operations on these frames. The closing operation will fill out the noise which is present in the crowd. So false detection can be reduced.<br>
<br>
<br>
<p align="center">
<img src="/images/code4.png" height=300 width=700 />
</p>
<br>
After the closing operations the frames look like this:<br>
<br>
<br>
<p align="center">
<img src="/images/code4_op.png" height=400 width=600 />
</p>
<br>

The contours are a useful tool for shape analysis and object detection and recognition. We find contours on every frame. Hence to detect players we check for contours whose height is greater than the width. We will do the masking operation on the detected players for detecting their jersey colour, if the jersey colour is blue, we will put text as “France” and if it is red then we will put text as “Belgium”. 

<br>
<br>
<p align="center">
<img src="/images/code5.png" height=400 width=600 />
</p>
<br>

We then detect the football using the same masking operations and save the images.<br>
<br>
<br>
<p align="center">
<img src="/images/code6.png" height=400 width=600 />
</p>
<br>

The output video and the entire code is uploaded on the google drive.
https://drive.google.com/drive/folders/1nJaoCdgIvOf_PZ33wjuZTFd8fKklZ6MU?usp=sharing.
