# <ins>Hand Gesture Recognition</ins>
The goal of our project was to analyze and correctly identify hand gesture movements collected by the Thalmic Labs Myo Gesture Control Armband sensor. The Myo Armband is used to record Surface Electromyography (sEMG) data. sEMG reads muscular impulses and is used in medical research. Thus, when attempting to classify this type of data, it is important to be as accurate as possible. Our desire is to achieve an accuracy that is sufficient enough to allow this data and other similar data to be used confidently for medical purposes. 

### How It Works

[![How It Works](https://i.imgur.com/Jy2RIpw.jpg)]()

### Data
The modified Boston housing dataset consists of 8961 data points, with each datapoint having 8 features. Data was collected for 22 pinch gesture subjects and 10 roshambo gesture subjects. Each person performed their entire set of hand gestures 3 times, 2 seconds per gesture, separated by a 1 second halt period, labeled as 'None'. During this intermission, no classified gesture was recorded. Each 2 second long gesture corresponded to a sequence of between 150 and 400 data points, each representing the sensor value at that interval.

The original dataset can be found [here](https://figshare.com/articles/EMG_from_forearm_datasets_for_hand_gestures_recognition/8666591/1).

### Installation
This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [TensorFlow](https://www.tensorflow.org)


In a terminal or command window, navigate to the project directory `src/` and run the following commands:

        python3 main.py

If missing libraries, install with:

        pip3 install [name]
        
## Built With

- Python

## Authors

- Sehej Sohal
- Daniel Loi
- James Garrett
