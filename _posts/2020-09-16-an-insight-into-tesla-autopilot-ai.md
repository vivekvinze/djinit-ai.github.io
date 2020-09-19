---
layout:     post
title:      An Insight into Tesla Autopilot AI
date:       2020-08-16 08:19:29
summary:    An Insight into Tesla Autopilot AI
categories: python, deep learning
---
## An Insight into Tesla Autopilot AI
<p align="center">
  <img src="/images/an-insight-tesla-autopilot-ai-2.png" />
</p>
Tesla Autopilot is a suite of advanced driver-assistance system features offered by Tesla that has lane centering, traffic-aware cruise control, self-parking, automatic lane changes, semi-autonomous navigation on limited access freeways, and the ability to summon the car from a garage or parking spot. In all of these features, the driver is responsible and the car requires constant supervision. The company claims the features reduce accidents caused by driver negligence and fatigue from long-term driving.

Whoa! That’s a lot of heavy-tech words.
To fully appreciate the awesomeness of this technological advancement made by Tesla, we must be acquainted with the concepts and challenges that define Autonomous Driving or rather that make a Driverless Car.

### What is a driverless car?
A driverless car, also known as an autonomous vehicle or a self-driving car, is a vehicle that is capable of sensing its environment and moving safely with little or no human input.
Self-driving cars combine a variety of sensors to perceive their surroundings, such as radar, lidar, sonar, GPS, odometry and inertial measurement units. Advanced control systems interpret sensory information to identify appropriate navigation paths, as well as obstacles and relevant signage.

### Levels of Autonomy in a DV:

<style>
#mytable {
  font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
  border-collapse: collapse;
  width: 100%;
}

#mytable td, #mytable th {
  border: 1px solid #ddd;
  padding: 8px;
}

#mytable tr:nth-child(even){background-color: #f2f2f2;}

#mytable tr:hover {background-color: #ddd;}

#mytable th {
  padding-top: 12px;
  padding-bottom: 12px;
  text-align: left;
  background-color: #4CAF50;
  color: white;
}
</style>

<table id="mytable">
  <tr>
    <th>Level &nbsp;</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><u><b>Level 0</b></u></td>
    <td>No Automation - This describes your everyday car.</td>
  </tr>
  <tr>
    <td><u><b>Level 1</b></u></td>
    <td>Driver Assistance - Here we can find your adaptive cruise control and lane keep assist to help with driving fatigue. Adaptive cruise control will keep a safe distance between you and the car ahead of you by using a combination of sensors, actuators and programs.</td>
  </tr>
  <tr>
    <td><u><b>Level 2</b></u></td>
    <td>Partial Automation - the driver must have hands on the wheel and be ready to take control at any given moment, level 2 automation can assist in controlling speed and steering.</td>
  </tr>
  <tr>
    <td><u><b>Level 3</b></u></td>
    <td>Conditional Automation - These autonomous vehicles are capable of driving themselves, but only under ideal conditions and with limitations, such as limited-access divided highways at a certain speed. Although hands are off the wheel, drivers are still required behind the wheel.</td>
  </tr>
  <tr>
    <td><u><b>Level 4</b></u></td>
    <td>High Automation - Autonomous vehicles that fall in this category can drive themselves without human interactions (besides entering your destination) but will be restricted to known use cases.</td>
  </tr>
  <tr>
    <td><u><b>Level 5</b></u></td>
    <td>Full Automation - Super Pursuit Mode! At Level 5 autonomy we arrive at true driverless cars. Level 5 capable vehicles should be able to monitor and maneuver through all road conditions and require no human interventions whatsoever, eliminating the need for a steering wheel and pedals.</td>
  </tr>
</table>

### Stages/Nodes of an Autonomous Vehicle:
<u><b>1.	PERCEPTION:</b></u> To recognize, detect and perceive it environment and detect changes or maybe obstacles in real-time. <br>
<u><b>2.	ESTIMATION:</b></u> To extrapolate the various vital parameters that govern the functioning of a self-driving car and also make appropriate calculations that will be used to maneuver the car without any hiccups. <br>
<u><b>3.	CONTROL:</b></u> To make decisions, and give control commands to actuators of the vehicle based on the data provided by perception and estimation nodes and also take past learning into account.<br>
<u><b>4.	CONTINUOUS INTEGRATION:</b></u> To interface all the nodes and monitor their working. Doing all that, putting together a car that is capable of safely and smoothly transporting the passenger or the goods aboard to the destination.<br>
<center><u>Here’s a typical flowchart of how a Tesla car drives on Autopilot</u></center>

<p align="center">
<img src="/images/an-insight-tesla-autopilot-ai-4.png"/>
</p>

### What are the challenges faced by a self-driving car?
The current technology is not fully capable of producing a level 4 or 5 DV because of many shortcomings.
•	The sensors that are equipped with latest cutting-edge technology may be successful in correctly perceiving the surrounding environment but are either not efficient for mass production or are not feasible economically.
•	There are a variety of conditions that a vehicle comes across on the road, some extraordinary enough that cannot be thought of before.
•	Even though a driver may come across most of the situations in his lifetime, gathering such data of the experiences of a myriad of cars is close to impossible with the current technology.
•	Considering all the above issues are sorted by some means, training an AI so powerful and thorough that adapts to all the features is very far-fetched. 
The engineers of today have found numerous techniques, algorithms and models to provide efficient and complete learning. But to use these models on such vast zeta-bytes of data requires very humongous computational power. The hardware and electronics today will be incapable to perform these operations in finite time.

### Why is TESLA Autopilot AI so popular?
Cause, it provides a near perfect solution to each of the shortcomings faced by a driverless car and is working towards getting better or rather the best.

### How is TESLA doing it?

<p align="center">
  <img src="/images/an-insight-tesla-autopilot-ai-5.png" />
</p>

Well, we are going to see each of its mind-blowingly phenomenal solutions in brief detail.

### SENSORS AND HARDWARE:
According To reports, TESLA has succeeded in building, testing and applying excellent sensors, and their peripheral hardware with reliable technologies like CAN 	bus that successfully deliver accurate measurements of important parameters with negligible time lag, so that the control system can make control decisions in time.
For perception, a Tesla car is equipped with a variety of LIDAR sensors, 5 RADARs, a range of Ultrasonic sensors. All of these are mated to a microcomputer of freaking high computational power. This astonishingly high computational power is achieved by a GPU with insane capabilities. All of this put together is Tesla Vision.

<p align="center">
  <img src="/images/an-insight-tesla-autopilot-ai-6.png" />
</p>

### TESLA’S SYSTEM ON CHIP COMPUTER:
That’s the chip Tesla uses in its cars to do the heavy duty AI computation
Power consumption: 100 W
Capable of 144 Trillion operations per second

<p align="center">
  <img src="/images/an-insight-tesla-autopilot-ai-7.png" />
</p>

This chip is perfect and efficient for tesla’s needs i.e. Fast, Cheap, Efficient and Powerful option with low power consumption for full autonomy, GROUND UP neural networks accelerator. The alternatives consume too much power and are bulky.
It has two independent computers (lighter blue and darker blue) which boot up separately.
These two computers introduce a redundancy, even if one fails the other will keep the car driving.
Elon Musk has claimed that the probability of a human falling unconscious during driving is higher than the probability of the system failing entirely.

### DATA ACQUISITION:
According to a report by Forbes, Tesla has long ago gathered over 2 billion miles of driving data that is used to train driverless cars. It will soon reach the 3 billion milestone. In comparison to its rivals, Tesla has taken a wildly massive lead in terms of collecting data which is one of the reasons Tesla has had kind of a breakthrough in self driving. Just for comparison, Waymo, Tesla’s biggest rival in autonomous locomotives industry, has gathered a mere 15 million miles of data, which may not be less but compared to Tesla is laughable. Tesla has used <u><i>Shadow Mode</i></u> for this purpose.

### SHADOW MODE:
This is another of Tesla’s brilliant innovations, where even while a car is in manual mode i.e. while it is being driven by a person, Tesla is constantly observing, recording, learning and simulating driverless driving. This unique variation of reinforcement learning helps it prepare a very efficient and realistic model that is prepared for real-life and real-time operations.
Another feature Tesla is testing right now is a predictive behavior as to what a pedestrian or cyclist in front of a vehicle might do. Then it will further predict to how all the vehicles around will react and move around giving it complete driving and road sense that even humans take great time, effort and experience to develop.

#### Click on image below to view the video
[![Watch the video](https://i.vimeocdn.com/video/603457588.webp?mw=1300&mh=731&q=70)](https://player.vimeo.com/video/192179726?app_id=122963)

### ADAPTIVE NEURAL NETWORKS!!!
A recent patent publication titled 'System and Method for Adapting a Neural Network Model on a Hardware Platform' has provided insight into the way that Tesla aims to create adaptable neural networks that can be used for various hardware platforms.
The patent application is a product of Tesla's acquisition of DeepScale, an AI startup that develops solutions for Full Self Driving as well as neural networks for small devices.
Generally speaking, neural network algorithms are trained to carry out a singular task with incredible efficacy. While they allow for patterns to be recognized in data at a rate that humans simply could never achieve, adapting them is often time-consuming for developers.
Tesla's answer to optimizing the adaptability of their neural networks, according to the recent patent filing, is to try to train and automate the sets of algorithms to be adaptable.
According to the patent filing, after plugging a neural network model and specific hardware information, software code analyzes the neural network and pinpoints where the decision points are. The system then runs hardware parameters against that information before providing available configurations.
<p align="center">
<img src="/images/an-insight-tesla-autopilot-ai-10.png" />
<img src="/images/an-insight-tesla-autopilot-ai-11.png" />
<img src="/images/an-insight-tesla-autopilot-ai-12.png" />
<img src="/images/an-insight-tesla-autopilot-ai-13.png" />
<img src="/images/an-insight-tesla-autopilot-ai-14.png" />
<img src="/images/an-insight-tesla-autopilot-ai-15.png" />
</p>

<p align="center">
Because of all these inventions, innovations, and breakthroughs Tesla is the biggest and undisputed leader of the autonomous driving industry! Think like Tesla!
</p>

<p align="center">
  <img src="/images/an-insight-tesla-autopilot-ai-16.png" />
</p>

<p align="center">A tribute to the greatest engineer of all time! </p>
