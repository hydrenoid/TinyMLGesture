from mpu6050 import mpu6050
import time
import csv
import keyboard

mpu = mpu6050(0x68)


def collect_data():
    with open('accelerometer_data.csv', 'a', newline='') as file:  # 'a' mode to append to the file
        writer = csv.writer(file)
        writer.writerow(["AX", "AY", "AZ", "GX", "GY", "GZ", "Label"])  # Header row

        while True:
            command = input("Type 'start' to begin recording, 'quit' to exit: ")
            if command.lower() == 'quit':
                break
            elif command.lower() == 'start':
                gesture_label = input("Enter the gesture label: ")
                print("Recording... Press Enter to stop.")
                while True:
                    accel_data = mpu.get_accel_data()
                    print("Acc X : " + str(accel_data['x']))
                    print("Acc Y : " + str(accel_data['y']))
                    print("Acc Z : " + str(accel_data['z']))
                    print()

                    gyro_data = mpu.get_gyro_data()
                    print("Gyro X : " + str(gyro_data['x']))
                    print("Gyro Y : " + str(gyro_data['y']))
                    print("Gyro Z : " + str(gyro_data['z']))
                    print()
                    print("-------------------------------")

                    Ax = accel_data['x']
                    Ay = accel_data['y']
                    Az = accel_data['z']

                    Gx = gyro_data['x']
                    Gy = gyro_data['y']
                    Gz = gyro_data['z']

                    writer.writerow([Ax, Ay, Az, Gx, Gy, Gz, gesture_label])
                    time.sleep(0.1)  # Adjust this based on your desired sampling rate

                    if keyboard.is_pressed('enter'):  # Check if Enter is pressed
                        print("Stopping recording...")
                        break


if __name__ == "__main__":
    collect_data()
