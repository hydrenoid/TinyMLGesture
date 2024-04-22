from mpu6050 import mpu6050
import time
import csv
import keyboard
import matplotlib.pyplot as plt
from drawnow import drawnow
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
from joblib import load

# Load in the model to be used for classification
model = load('gesture_classifier.joblib')

# Set system variables
mpu = mpu6050(0x68)

accel_x, accel_y, accel_z, accel_mag = [], [], [], []
gesture_data = []
gestures = ['point', 'raise-hand', 'dab', 'hair-swipe', 'rps']
target_duration = 1 / 90


# Takes in the file path and returns a list of the calculated features
def extract_features(file_path):
    df = pd.read_csv(file_path)
    features = {}
    for axis in ['x', 'y', 'z', 'mag']:
        features[f'{axis}_mean'] = df[axis].mean()
        features[f'{axis}_std'] = df[axis].std()
        features[f'{axis}_max'] = df[axis].max()
        features[f'{axis}_min'] = df[axis].min()
        features[f'{axis}_kurtosis'] = df[axis].kurtosis()
        features[f'{axis}_skew'] = df[axis].skew()
    return features

# Plot configuration for not recording
def make_fig():
    plt.ylim(-20, 20)  # Set the y-axis limits
    plt.title('Real-Time Accelerometer Data')
    plt.grid(True)
    plt.ylabel('Acceleration')
    plt.plot(accel_x, 'r--', label='X-axis')
    plt.plot(accel_y, 'g--', label='Y-axis')
    plt.plot(accel_z, 'b--', label='Z-axis')
    plt.plot(accel_mag, 'c--', label='mag')
    plt.legend(loc='upper left')

# Plot configuration for recording
def make_fig_recorded():
    plt.ylim(-20, 20)  # Set the y-axis limits
    plt.title('Real-Time Accelerometer Data')
    plt.grid(True)
    plt.ylabel('Acceleration')
    plt.plot(accel_x, 'r-', label='X-axis')
    plt.plot(accel_y, 'g-', label='Y-axis')
    plt.plot(accel_z, 'b-', label='Z-axis')
    plt.plot(accel_mag, 'c-', label='magnitude')
    plt.legend(loc='upper left')


def main():
    # Check if the user would like to start
    if input("Would you like to start? (yes/no) ").lower() == 'yes':
        while True:
            # Begin displaying live data here
            # Ask user to press button to record gesture then output the calculated result
            print("Press enter to begin recording the gesture.")
            while True:
                # Track time for consistent recording
                start_time = time.time()

                # Read data from accelerometer
                accel_data = mpu.get_accel_data()

                # Append new data to the lists
                accel_x.append(accel_data['x'])
                accel_y.append(accel_data['y'])
                accel_z.append(accel_data['z'])

                x = accel_data['x']
                y = accel_data['y']
                z = accel_data['z']

                # Calculate magnitude
                mag = math.sqrt((x * x) + (y * y) + (z * z))
                accel_mag.append(mag)

                # Update the plot
                drawnow(make_fig)

                # Limit the size of the lists to prevent memory issues
                if len(accel_x) > 50:
                    del accel_x[0]
                    del accel_y[0]
                    del accel_z[0]
                    del accel_mag[0]

                # If user presses enter they will then move on to the recording loop
                if keyboard.is_pressed('enter'):
                    break

                # See how long it has been for this iteration for frequency and consistency
                elapsed = time.time() - start_time  # Calculate elapsed time
                sleep_time = target_duration - elapsed

                if sleep_time > 0:
                    time.sleep(sleep_time)  # Sleep to maintain approximately 90 Hz frequency

            # Countdown from 3 before starting recording
            for i in range(3, 0, -1):
                print(i)
                time.sleep(1)

            print('GOOO!!!')

            # After recording, check if button is hit to signal that the gesture is done
            print("Press 'enter' to indicate the gesture is done.")
            test_counter = 0  # this counter is to prevent excessively long recordings

            while True:
                # Track time for consistent recording frequency
                start_time = time.time()

                accel_data = mpu.get_accel_data()

                x = accel_data['x']
                y = accel_data['y']
                z = accel_data['z']
                mag = math.sqrt((x * x) + (y * y) + (z * z))

                # add accelerometer data to be recorded
                gesture_data.append([x, y, z, mag])

                # Append new data to the lists
                accel_x.append(accel_data['x'])
                accel_y.append(accel_data['y'])
                accel_z.append(accel_data['z'])
                accel_mag.append(mag)

                # Update the plot
                drawnow(make_fig_recorded)

                # Limit the size of the lists to prevent memory issues
                if len(accel_x) > 50:
                    del accel_x[0]
                    del accel_y[0]
                    del accel_z[0]
                    del accel_mag[0]

                # If keyboard is pressed stop recording and save data
                if keyboard.is_pressed('enter'):  # Check if Enter is pressed
                    print("Stopping recording...")
                    time.sleep(0.2)
                    # Added this loop as sometimes it would not register if you pressed enter quickly
                    while keyboard.is_pressed('enter'):
                        time.sleep(0.1)
                    break
                # No gesture should be greater than 600 as it will be too long (prevents crashing and errors)
                elif test_counter > 600:
                    break

                elapsed = time.time() - start_time  # Calculate elapsed time
                sleep_time = target_duration - elapsed

                if sleep_time > 0:
                    time.sleep(sleep_time)  # Sleep to maintain approximately 90 Hz frequency

                test_counter = test_counter + 1

            # Stop recording and run model prediction, then print the calculated gesture
            file_name = 'recorded_data.csv'
            with open(file_name, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(["x", "y", "z", 'mag'])
                for item in gesture_data:
                    writer.writerow(item)

            # extract features from recorded data
            extracted_features = extract_features(file_name)
            features_df = pd.DataFrame([extracted_features])

            # make prediction
            predicted_gesture = model.predict(features_df)
            print("Predicted Gesture:", predicted_gesture[0])

            # clear recorded data
            gesture_data.clear()


# Run the program
if __name__ == "__main__":
    main()
