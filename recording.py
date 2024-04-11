from mpu6050 import mpu6050
import time
import csv
import keyboard
import matplotlib.pyplot as plt
from drawnow import drawnow
import math

mpu = mpu6050(0x68)

accel_x, accel_y, accel_z = [], [], []
gesture_data = []
gestures = ['wave', 'gritty', 'forehand-tennis', 'backhand-tennis', 'handshake']
target_duration = 1 / 90

def make_fig():
    plt.ylim(-20, 20)  # Set the y-axis limits
    plt.title('Real-Time Accelerometer Data')
    plt.grid(True)
    plt.ylabel('Acceleration')
    plt.plot(accel_x, 'r-', label='X-axis')
    plt.plot(accel_y, 'g-', label='Y-axis')
    plt.plot(accel_z, 'b-', label='Z-axis')
    plt.legend(loc='upper left')

def main():
    # Ask user for their name to create a file name
    user_name = input("Enter your name for the file name: ")
    file_name = user_name + "_data.csv"  # You might want to add more detail to the file naming

    # Try opening the file, handle potential errors with try/except block
    try:
        with open(file_name, 'w') as file:
            # Check if the user would like to start
            if input("Would you like to start? (yes/no) ").lower() == 'yes':
                # Begin displaying live data here (Placeholder for actual data display logic)
                writer = csv.writer(file)

                # Iterate recording for each gesture
                for gesture in gestures:
                    for count in range(5):
                        # Display data and prompt user to hit button to begin recording
                        print('The current gesture is: ' + gesture + '.')
                        print('You have ' + (5 - count) + ' entries left.')
                        print("Press enter to begin recording the gesture.")
                        while True:
                            start_time = time.time()

                            accel_data = mpu.get_accel_data()

                            # Append new data to the lists
                            accel_x.append(accel_data['x'])
                            accel_y.append(accel_data['y'])
                            accel_z.append(accel_data['z'])

                            # Update the plot
                            drawnow(make_fig)

                            if keyboard.is_pressed('enter'):
                                break

                            elapsed = time.time() - start_time  # Calculate elapsed time
                            sleep_time = target_duration - elapsed

                            if sleep_time > 0:
                                time.sleep(sleep_time)  # Sleep to maintain approximately 90 Hz frequency
                            else:
                                print("Warning: Loop iteration took longer than target duration")

                        # Countdown from 3 before starting recording
                        for i in range(3, 0, -1):
                            print(i)
                            time.sleep(1)

                        # After recording, check if button is hit to signal that the gesture is done
                        print("Press 'b' to indicate the gesture is done.")
                        while True:
                            start_time = time.time()

                            accel_data = mpu.get_accel_data()
                            print("Acc X : " + str(accel_data['x']))
                            print("Acc Y : " + str(accel_data['y']))
                            print("Acc Z : " + str(accel_data['z']))
                            print()
                            print("-------------------------------")

                            x = accel_data['x']
                            y = accel_data['y']
                            z = accel_data['z']
                            mag = math.sqrt((x * x) + (y * y) + (z * z))

                            gesture_data.append([x, y, z, mag, gesture])

                            # Append new data to the lists
                            accel_x.append(accel_data['x'])
                            accel_y.append(accel_data['y'])
                            accel_z.append(accel_data['z'])

                            # Update the plot
                            drawnow(make_fig)

                            # Limit the size of the lists to prevent memory issues
                            if len(accel_x) > 50:
                                del accel_x[0]
                                del accel_y[0]
                                del accel_z[0]

                            if keyboard.is_pressed('enter'):  # Check if Enter is pressed
                                print("Stopping recording...")
                                time.sleep(0.2)
                                while keyboard.is_pressed('enter'):
                                    time.sleep(0.1)

                                break  # Sleep to prevent high CPU usage

                            elapsed = time.time() - start_time  # Calculate elapsed time
                            sleep_time = target_duration - elapsed

                            if sleep_time > 0:
                                time.sleep(sleep_time)  # Sleep to maintain approximately 90 Hz frequency
                            else:
                                print("Warning: Loop iteration took longer than target duration")

                        # Stop recording and save data to file (Placeholder for actual save logic)
                        writer.writerow(["x", "y", "z", 'mag', gesture])
                        for item in gesture_data:
                            writer.writerow(item)

            else:
                # User chose not to start, so we close the file and stop displaying data
                print("Exiting the program.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # The program reaches the end
    print("Program has ended. Goodbye!")

# Run the program
if __name__ == "__main__":
    main()
