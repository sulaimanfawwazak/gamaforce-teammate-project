#! /usr/bin/env python3
import time
import rospy
from tf.transformations import quaternion_from_euler
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from geometry_msgs.msg import TwistStamped
from simple_pid import PID
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
import cv2 as cv
import threading
import numpy as np
import math

# Class for Flight Modes
class FCUModes():
    def __init__(self):
        self.height = 3

    def setTakeoff(self):
        rospy.wait_for_service("mavros/cmd/takeoff")
        try:
            takeoffService = rospy.ServiceProxy("mavros/cmd/takeoff", CommandTOL)
            takeoffService(altitude=self.height)
            rospy.loginfo(f"Takeoff {self.height} meter")

        except rospy.ServiceException as e:
            rospy.logerr("Takeoff Failed")

    def setArm(self):
        rospy.wait_for_service("mavros/cmd/arming")
        try:
            armService = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
            armService(True)
            rospy.loginfo("Arming")

        except rospy.ServiceException as e:
            rospy.logerr("Arming Failed")

    def setDisarm(self):
        rospy.wait_for_service("mavros/cmd/arming")
        try:
            armService = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
            armService(False)
            rospy.loginfo("Disarming")

        except rospy.ServiceException as e:
            rospy.logerr("Arming Failed")

    def setGuided(self):
        rospy.wait_for_service("mavros/set_mode")
        try:
            guidedService = rospy.ServiceProxy("mavros/set_mode", SetMode)
            guidedService(custom_mode="GUIDED")
            rospy.loginfo("Entering GUIDED Mode")

        except rospy.ServiceException as e:
            rospy.logerr("Guided Failed")

    def setLand(self):
        rospy.wait_for_service("mavros/set_mode")
        try:
            landService = rospy.ServiceProxy("mavros/set_mode", SetMode)
            landService(custom_mode="LAND")
            rospy.loginfo("landing")
        except rospy.ServiceException as e:
            rospy.logerr("Landing failed")

# Class for PID Controller
class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.setpoint = setpoint

        self.pid = PID(kp, ki, kd, setpoint=setpoint)

        # Initialize the error & integral to 0
        self.prev_error = 0
        self.integral = 0

    def compute(self, currentValue):
        error = self.setpoint - currentValue
        self.integral += error
        derivative = error - self.prev_error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Updates the error
        self.prev_error = error

        return output
    
class Controller:
    def __init__(self):
        self.pub_velocity = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        self.state = State()
        self.bridge = CvBridge()
        self.cv_image = None
        # self.cv_image = cv.cvtColor(self.cv_image, cv.COLOR_BGR2HSV)
        # self.cv_image = rospy.Subscriber('/webcam2/image_raw2', Image, callback=self.camCallback)
        self.pid_controller = PIDController(kp=5, ki=0.1, kd=10, setpoint=0.0)
        rospy.Subscriber('/mavros/state', State, callback=self.stateCallback)
        rospy.Subscriber('/webcam2/image_raw2', Image, callback=self.camCallback)
        self.pub_cam2 = rospy.Publisher('/webcam2/image_raw2/processed', Image, queue_size=10)
        self.mask = None

        self.isBoxDetected = False


    def moveAtVelocity(self, lin_x, lin_y, lin_z, ang_x, ang_y, ang_z):
        pub_data = TwistStamped()

        pub_data.twist.linear.x = lin_x
        pub_data.twist.linear.y = lin_y
        pub_data.twist.linear.z = lin_z
        pub_data.twist.angular.x = ang_x
        pub_data.twist.angular.y = ang_y
        pub_data.twist.angular.z = ang_z

        rospy.loginfo(f'Travel at {lin_y} m/s')

        self.pub_velocity.publish(pub_data)
        # time.sleep(3)
        # await asyncio.sleep(1)
            
    def brake(self, target_velocity):
        rospy.logdebug('Start braking')
        pub_data = TwistStamped()

        # pub_data.twist.linear.x = 0
        # pub_data.twist.linear.y = 0
        # pub_data.twist.linear.z = 0
        # pub_data.twist.angular.x = 0
        # pub_data.twist.angular.y = 0
        # pub_data.twist.angular.z = 0
        # 
        # self.pub_velocity.publish(pub_data)
        self.moveAtVelocity(0, 0, 0, 0, 0, 0)

        current_velocity = target_velocity

        while abs(target_velocity - current_velocity) > 0.1:
            currentVelocity = target_velocity - self.pidController.compute(target_velocity)

            pub_data.twist.linear.x = 0
            pub_data.twist.linear.y = currentVelocity
            pub_data.twist.linear.z = 0
            pub_data.twist.angular.x = 0
            pub_data.twist.angular.y = 0
            pub_data.twist.angular.z = 0

            self.pub_velocity.publish(pub_data)

    def detectBox(self):
        green = np.array([35, 96, 96])

        self.mask = cv.inRange(self.cv_image, green, green)

        contours, _ = cv.findContours(self.mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            rospy.logdebug('Box Detected !!!')
            self.isBoxDetected = True
        else:
            rospy.logerr('Box Not Detected')
            # self.moveAtVelocity(0, 20, 0, 0, 0, 0)

        self.pub_cam_data = self.bridge.cv2_to_imgmsg(self.mask, encoding="passthrough")
        # time.sleep(0.5)
        self.pub_cam2.publish(self.pub_cam_data)

        # await asyncio.sleep(0.1)
        
        # return self.isBoxDetected
        
    def camCallback(self, msg):
        # self.cv_image = msg
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # if len(self.cv_image.shape) == 3 and self.cv_image.shape[2] == 3:
        #     # Convert the image to grayscale
        #     self.cv_image = cv.cvtColor(self.cv_image, cv.COLOR_BGR2GRAY)

    def stateCallback(self, msg):
        self.state = msg

    def runDetectBoxLoop(self):
        while not self.isBoxDetected and not rospy.is_shutdown():
            self.detectBox()
            time.sleep(0.1)

    def runMoveLoop(self):
        while not self.isBoxDetected and not rospy.is_shutdown():
            self.moveAtVelocity(0, 20, 0, 0, 0, 0)
            time.sleep(0.6)

    def detectGapura(self):
        # Make the crosshair
        self.tinggi_citra, self.lebar_citra, _ = self.cv_image.shape
        self.crosshair_x = self.lebar_citra // 2
        self.crosshair_y = self.tinggi_citra // 2
        self.crosshair = (self.crosshair_x, self.crosshair_y)

        cv.line(self.cv_image, (self.crosshair[0], self.crosshair[1] + 10), (self.crosshair[0], self.crosshair[1] - 10), (0, 255, 255), 1)
        cv.line(self.cv_image, (self.crosshair[0] + 10, self.crosshair[1]), (self.crosshair[0] - 10, self.crosshair[1]), (0, 255, 255), 1)

        # cv.line(self.cv_image, self.crosshair_y[1] + (10, 10), self.crosshair_y - (10, 10), (0, 255, 255), 1) 

        # Convert from BGR to RGB
        self.cv_image_rgb = cv.cvtColor(self.cv_image, cv.COLOR_BGR2RGB)

        # Process the image to make mask
        gapura = np.array([71, 0, 0])
        red_mask = cv.inRange(self.cv_image_rgb, gapura, gapura)
        kernel = np.ones((5, 5), np.uint8) 
        red_mask = cv.dilate(red_mask, kernel, iterations=2)

        # Find the contours
        contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Find the object with the largest contour area
        largest_object = None
        largest_area = 0

        for contour in contours:
            area = cv.contourArea(contour)
            if area > largest_area:
                largest_object = contour
                largest_area = area

        result_image = np.zeros_like(red_mask)

        if largest_object is not None:
            contour = cv.drawContours(result_image, [largest_object], -1, 255, thickness=cv.FILLED)
            cv.imshow('Segmented Image', result_image)
        
        # Find the centroids/moments
        _, _, _, centroids = cv.connectedComponentsWithStats(result_image, connectivity=4)
        self.moment_x, self.moment_y = centroids[0]
        self.moment_x = int(self.moment_x)
        self.moment_y = int(self.moment_y)

        cv.circle(self.cv_image_rgb, (self.moment_x, self.moment_y), 4, (0, 255, 0), -1)

        return self.crosshair_x, self.crosshair_y, self.moment_x, self.moment_y, 

    def calculate_angle(x1, y1, x2, y2):
        delta_x = x2 - x1
        delta_y = y2 - y1

        degrees = np.degrees(np.arctan2(delta_y, delta_x))
        return degrees
    
    def calculate_distance(self, x1, y1, x2, y2):
        delta_x = x2 - x1
        delta_y = y2 - y1

        return delta_x, delta_y
    
    # def centeringWithAngle(self):
    #     # Tolerance
    #     tolerance = np.degrees(3)

    #     crosshair_x, crosshair_y, moment_x, moment_y = self.detectGapura()

    #     # Calculate the angle between crosshair and moment
    #     angle = self.calculate_angle(crosshair_x, crosshair_y, moment_x, moment_y)

    #     # while abs(angle) > tolerance:
    #     #     if current_time - centering_time_start < 5:
    #     #         self.moveAtVelocity()

    def centeringWithDistance(self):
        # Distance Tolerance
        tolerance = 2

        crosshair_x, crosshair_y, moment_x, moment_y = self.detectGapura()

        # Calculate the distance
        delta_x, delta_y = self.calculate_distance(crosshair_x, crosshair_y, moment_x, moment_y)

        while abs(delta_x) > tolerance:
            # Move to the right if delta_x is positive
            if delta_x > 0:
                rospy.loginfo('Centering ...')
                self.moveAtVelocity(0.5, 0, 0, 0 ,0 ,0)
            
            # Move to the left if delta_x is negative
            elif delta_x < 0:
                rospy.loginfo('Centering ...')
                self.moveAtVelocity(-0.5, 0, 0, 0, 0 ,0)

            # Recalculate the distance
            delta_x, delta_y = self.calculate_distance(crosshair_x, crosshair_y, moment_x, moment_y)
        
        rospy.logdebug('Centering Complete !!!')

        centering_time_start = time.time()
        centering_time_elapsed = 0

        # Wait for 5 seconds after centering
        while abs(delta_x) < tolerance and centering_time_elapsed < 5:
            centering_time_end = time.time()
            centering_time_elapsed = centering_time_end - centering_time_start

        move_time_start = time.time()
        move_time_elapsed = 0

        # Move for 2 seconds at 5 m/s
        while move_time_elapsed < 4:
            self.moveAtVelocity(0, 5, 0, 0, 0, 0)
            move_time_end = time.time()
            move_time_elapsed = move_time_end - move_time_start
            
def main():
    modes = FCUModes()
    control = Controller()

    rospy.init_node('Copter', anonymous=True)

    rate = rospy.Rate(10)

    # Go to GUIDED mode
    modes.setGuided()

    while not control.state.armed:
        modes.setArm()
        rate.sleep()
    
    if control.state.armed:
        rospy.sleep(5)

        # Takeoff
        modes.setTakeoff()
        rospy.sleep(8)

    rospy.logdebug("Takeoff Success!!!")

    # Gapura 1
    control.moveAtVelocity(-2, 0, 0, 0 , 0, 0)
    rospy.sleep(5)
    control.detectGapura()
    rospy.sleep(5)
    control.centeringWithDistance()
    control.brake(0)

    # Gapura 2
    rospy.sleep(5)
    control.moveAtVelocity(2, 0, 0, 0, 0, 0,)
    rospy.sleep(5)
    control.detectGapura()
    rospy.sleep(5)
    control.centeringWithDistance()
    control.brake(0)    

    # Threads for running in different pace
    detect_thread = threading.Thread(target=control.runDetectBoxLoop)
    move_thread = threading.Thread(target=control.runMoveLoop)

    # Start the threads
    move_thread.start()
    detect_thread.start()

    while not control.isBoxDetected:
        time.sleep(0.1)

    # Stops the threads
    move_thread.join()
    detect_thread.join()

    # Starts the breaking
    control.brake(0)

    # Landing
    modes.setLand()
    rospy.sleep(5)

    # Disarming
    modes.setDisarm()
    # while True:
    #     control.detectBox()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass