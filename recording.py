from mpu6050 import mpu6050
import time
import csv
import keyboard
import matplotlib.pyplot as plt
from drawnow import drawnow
import math

mpu = mpu6050(0x68)

accel_x, accel_y, accel_z, accel_mag = [], [], [], []
gesture_data = []
gestures = ['point', 'raise-hand', 'forehand-tennis', 'backhand-tennis', 'rps']
target_duration = 1 / 90


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
    # Ask user for their name to create a file name
    user_name = input("Enter your name for the file name: ")

    # Check if the user would like to start
    if input("Would you like to start? (yes/no) ").lower() == 'yes':
        # Begin displaying live data here (Placeholder for actual data display logic)

        # Iterate recording for each gesture
        for gesture in gestures:
            for count in range(5):
                # Display data and prompt user to hit button to begin recording
                print('The current gesture is: ' + str(gesture) + '.')
                print('You have ' + str((5 - count)) + ' entries left.')
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
                    elif test_counter > 600:
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

                    gesture_data.append([x, y, z, mag, gesture])

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

                # Stop recording and save data to file (Placeholder for actual save logic)
                try:
                    file_name = user_name + '_' + gesture + str(count) + '_data.csv'
                    with open(file_name, 'w') as file:
                        writer = csv.writer(file)
                        writer.writerow(["x", "y", "z", 'mag', gesture])
                        for item in gesture_data:
                            writer.writerow(item)
                        gesture_data.clear()

                except Exception as e:
                    print(f"An error occurred: {e}")

    else:
        # User chose not to start, so we close the file and stop displaying data
        print("Exiting the program.")

    # The program reaches the end
    print("Program has ended. Goodbye!")


# Run the program
if __name__ == "__main__":
    main()
