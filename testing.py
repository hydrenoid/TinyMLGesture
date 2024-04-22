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

model = load('gesture_classifier.joblib')

mpu = mpu6050(0x68)

accel_x, accel_y, accel_z, accel_mag = [], [], [], []
gesture_data = []
gestures = ['point', 'raise-hand', 'dab', 'hair-swipe', 'rps']
target_duration = 1 / 90


def extract_features(df):
    features = {}
    for axis in ['x', 'y', 'z', 'mag']:
        features[f'{axis}_mean'] = df[axis].mean()
        features[f'{axis}_std'] = df[axis].std()
        features[f'{axis}_max'] = df[axis].max()
        features[f'{axis}_min'] = df[axis].min()
        features[f'{axis}_kurtosis'] = df[axis].kurtosis()
        features[f'{axis}_skew'] = df[axis].skew()
        # Add other features as per your model's training
    return features

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
        # Begin displaying live data here (Placeholder for actual data display logic)

        # Ask user to press button to record gesture then output the calculated result

        # Display data and prompt user to hit button to begin recording
        print("Press enter to begin recording the gesture.")
        test_counter = 0
        while True:

            start_time = time.time()

            accel_data = mpu.get_accel_data()

            # Append new data to the lists
            accel_x.append(accel_data['x'])
            accel_y.append(accel_data['y'])
            accel_z.append(accel_data['z'])

            x = accel_data['x']
            y = accel_data['y']
            z = accel_data['z']
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

            if keyboard.is_pressed('enter'):
                break

            elapsed = time.time() - start_time  # Calculate elapsed time
            sleep_time = target_duration - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)  # Sleep to maintain approximately 90 Hz frequency

            test_counter = test_counter + 1

        # Countdown from 3 before starting recording
        for i in range(3, 0, -1):
            print(i)
            time.sleep(1)

        print('GOOO!!!')

        # After recording, check if button is hit to signal that the gesture is done
        print("Press 'enter' to indicate the gesture is done.")
        test_counter = 0

        while True:
            start_time = time.time()

            accel_data = mpu.get_accel_data()

            x = accel_data['x']
            y = accel_data['y']
            z = accel_data['z']
            mag = math.sqrt((x * x) + (y * y) + (z * z))

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

            if keyboard.is_pressed('enter'):  # Check if Enter is pressed
                print("Stopping recording...")
                time.sleep(0.2)
                while keyboard.is_pressed('enter'):
                    time.sleep(0.1)

                break
            elif test_counter > 600:
                break

            elapsed = time.time() - start_time  # Calculate elapsed time
            sleep_time = target_duration - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)  # Sleep to maintain approximately 90 Hz frequency

            test_counter = test_counter + 1

        # Stop recording and run model prediction, then print the calculated gesture

        # extract features from recorded data
        extracted_features = extract_features(gesture_data)
        features_df = pd.DataFrame([extracted_features])

        # make prediction
        predicted_gesture = model.predict(features_df)
        print("Predicted Gesture:", predicted_gesture[0])

        # clear recorded data
        gesture_data.clear()


# Run the program
if __name__ == "__main__":
    main()
