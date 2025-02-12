import serial
import time

# Define the serial port and baud rate.
# Ensure the 'COM#' corresponds to what was seen in the Windows Device Manager
ser = serial.Serial('COM4', 9600)

def led_on_off():
    user_input = input("\n Type on / off / quit : ")
    if user_input =="on":
        print("temp sensor is on...")
        time.sleep(0.1) 
        ser.write(b'H') 
        led_on_off()
    elif user_input =="off":
        print("temp sensor is off...")
        time.sleep(0.1)
        ser.write(b'L')
        led_on_off()
    if user_input =="on":
        print("heartbeat sensor is on...")
        time.sleep(0.1) 
        ser.write(b'H') 
        led_on_off()
    elif user_input =="off":
        print("heartbeat sensor is off...")
        time.sleep(0.1)
        ser.write(b'L')
        led_on_off()
    elif user_input =="quit" or user_input == "q":
        print("Program Exiting")
        time.sleep(0.1)
        ser.write(b'L')
        ser.close()
    else:
        print("Invalid input. Type on / off / quit.")
        led_on_off()

time.sleep(2) # wait for the serial connection to initialize

led_on_off()