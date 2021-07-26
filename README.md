# Brain-Computer-Interface-Controller-
This project was aimed at classifying motor imagery from EEG data using machine learning.
Firstly I did some work using offline analysis and for online decoding I plan to train the 
on a few trials and fine tune a pre-trained program. The game will be whackamole!

## Whackamole
The user will be imagining left or right motor imagery and the decoder will output 1 or 0
based on presence of motor imagery. When the mole comes up they are to start big braining hard
and if a movement imagination is registered the hammer will strike the mole :P

## Methods for online and offline
I used both Power Spectral Density and Common Spatial Patterns to extract the EEG features

## Algorithms
Used a bunch of them, compared via confusion matrix

## Classes
The two class ones are just left and right hand
The three class ones are left, right and rest

## Example trial + online classification
https://user-images.githubusercontent.com/69587452/113508056-35c79180-9591-11eb-88aa-aa1a9b99cb60.mp4


