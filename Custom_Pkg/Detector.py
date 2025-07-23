#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 23:40:07 2024

@author: sepehr
"""

import cv2
from ultralytics import YOLO
import threading
from queue import Queue, Full, Empty
import numpy as np

#class to handle video processing

class VideoProcessor:
    
    #initializer method, constructor
    def __init__( self, video_path ):
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture( video_path )
        
        fps = self.cap.get( cv2.CAP_PROP_FPS )
        
        #variables used later for determining the delay based on playback speed
        self.normal_speed = int( 1000 / fps )
        
        self.half_speed = int( 2000 / fps )
        
        assert self.cap.isOpened(), "Error opening video file."
        
    #method to release video resources
    def release( self ):     
        
        self.cap.release()

    #method to read frame from the video 
    def get_frame( self ):
        
        ret, frame = self.cap.read()
        
        if not ret:
            
            return None
        
        return frame
    
    #method to define output video properties 
    def init_video_writer( self, output_path ):
        
        width = int( self.cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
        
        height = int( self.cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
        
        fps = self.cap.get( cv2.CAP_PROP_FPS )
        
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        
        return cv2.VideoWriter( output_path, codec, fps, (width, height) )
    
    def play_video( self, half_speed=False ):
        
        #reinitializing video capture if necessary
        
        if not self.cap.isOpened():
            
            self.cap = cv2.VideoCapture( self.video_path )
            
            assert self.cap.isOpened(), "Error opening video file."
    
        delay = self.half_speed if half_speed else self.normal_speed
    
        #display
        try:
            
            while True:
                
                frame = self.get_frame()
    
                if frame is None:
                    break
    
                cv2.imshow('Video', frame)
    
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            
            print( f"An error occurred: {e}" )
            
        finally:
            self.release()
            cv2.destroyAllWindows()


#VideoProcessor extended to MLDetector

class MLDetector( VideoProcessor ):

    #Initializer method
    def __init__( self, video_path, model_path ):
        
        #base class is initialized with the video path
        super().__init__(video_path)
        
        #load YOLO model
        self.model = YOLO( model_path )

    #process video and annotate cracks
    #Reference : 
        # https://docs.ultralytics.com/modes/predict/#streaming-source-for-loop

    def process_video( self, output_path ):
        
        out = self.init_video_writer( output_path )

        #process the video frames
        while self.cap.isOpened():  
            
            frame = self.get_frame()
            
            if frame is not None:
                
                #run YOLOv8 inference on frame
                
                results = self.model(frame)
                
                #visualize the result
                
                annotated_frame = results[0].plot()
                
                #write the annotated frame to the output file
                
                out.write( annotated_frame )
                
            else:
                
                #break the loop if the end of the video is reached
                break

        #release everything after job is finished
        
        self.release()
        
        out.release()
        
        cv2.destroyAllWindows()
        
#VideoProcessor extended to Canny Detector

class CannyDetector(VideoProcessor):
    
    def __init__( self, video_path, 
                 canny_threshold_low = 75,
                 canny_threshold_high = 175, 
                 min_contour_length= 275,
                 blur_size = 7, 
                 kernel_size = 3 ):
        
        super().__init__(video_path)
        
        #adjustable parameters for Canny edge detection
        
        self.canny_threshold_low = canny_threshold_low      
        self.canny_threshold_high = canny_threshold_high
        self.min_contour_length = min_contour_length
        self.blur_size =  self.make_odd(blur_size)
        self.kernel_size = self.make_odd(kernel_size)
        
    #ensure value is always odd
    def make_odd( self, value ):
        
        return value if value % 2 != 0 else value + 1

    def detect_cracks( self, frame ):
        
        if frame is None:
            return None

        #convert to grayscale
        
        gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )

        #apply a Gaussian blur
        
        blurred = cv2.GaussianBlur( gray, ( self.blur_size, self.blur_size ), 0 )

        #apply Canny edge detection with dynamic thresholds
        edges = cv2.Canny( blurred, 
                           self.canny_threshold_low, 
                           self.canny_threshold_high )

        """Apply morphological closure. Source : 
        https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        Closure is useful for removing noise"""
        
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        
        closure = cv2.morphologyEx( edges, cv2.MORPH_CLOSE, kernel )

        #find contours based on edges
        
        contours, _ = cv2.findContours( closure, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE )

        #filter contours based on length
        
        long_contours = [contour for contour in contours 
                         if cv2.arcLength(contour, True) > self.min_contour_length]

        #drawing contours on the original frame
        
        cv2.drawContours(frame, long_contours, -1, (0, 255, 0), 2)

        return frame
         
    #process Canny Edge output video
    def process_video( self, output_path ):
        
        out = self.init_video_writer(output_path)
        
        while self.cap.isOpened():
            
            frame = self.get_frame()
            
            if frame is not None:
                
                annotated_frame = self.detect_cracks(frame)
                
                if annotated_frame is None or annotated_frame.size == 0:
                    print("Annotated frame is empty or None.")
                    continue
                
                out.write(annotated_frame)
            else:
                break
            
        self.release()
        out.release()
        cv2.destroyAllWindows()

    
"""
CannyRealTime, is for real-time modification of processed frames. 
By utilizing multithreading and queues, I separated the task of reading and 
processing video frames from the main thread, which is responsible for 
displaying the processed frames. I introduced interactivity to adjust values.
"""

#Source: https://superfastpython.com/thread-queue/

#CannyDetector extends to CannyRealTime
    
class CannyRealTime(CannyDetector):
        
    def __init__(self, video_path):
        
        super().__init__(video_path)
        
        #queue for thread communication
        
        self.frame_queue = Queue(10)
                
        #initializing thread control
        
        self.read_thread = None          
        self.stop_thread = False
        self.no_more_frames = False

    
    #reads frames in a separate thread processing them for crack detection
    def frame_reader( self ):
        
        while not self.stop_thread:
            
            frame = self.get_frame()
            
            if frame is None:
                
                self.no_more_frames = True
                
                break
            
            processed_frame = self.detect_cracks( frame )
            
            #putting processed frame into queue
            
            try:
                
                self.frame_queue.put( processed_frame, timeout=1 )
                
            except Full:
                
                print("Queue is full, skipping this frame.")

        print("Frame_reader thread has exited.")
        
    
    #play video with processed frame
    #also method overriding                
    def play_video( self, half_speed=False ):
                        
        #create trackbars
        self.create_trackbars()

        #reset thread control flag
        self.stop_thread = False

        #start the frame reader thread
        self.read_thread = threading.Thread( target=self.frame_reader )
        self.read_thread.start()
        
        #the display loop
        while not self.no_more_frames:
            
            try:
                #retrieve processed frame from the queue
                processed_frame = self.frame_queue.get( timeout=1 )
                
            except Empty:
                #if the queue is empty, wait briefly and continue the loop
                cv2.waitKey(20) #for GUI events
                continue
                        
            #dynamically adjust trackbar
            self.update_trackbar_values(processed_frame)          
            
            delay = self.half_speed if half_speed else self.normal_speed
            
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                self.stop_thread = True
                break

        self.clean_up()
    
    #terminate threads and release resources
    def clean_up(self):
        
        print( "Initiating clean up..." )
        
        #setting the flag to signal frame_reader thread to stop
        self.stop_thread = True
    
        #check if frame_reader thread is still running
        if self.read_thread is not None:
            
            print( "Waiting for read thread to join..." )

            self.read_thread.join( timeout=2 )

            if self.read_thread.is_alive():
                
                print( "Warning! Read thread not terminated." )
    
        self.release()

        cv2.destroyAllWindows()

        print( "Clean up completed." )
    
    #interactive threshold adjustments
    
    #Source: https://docs.opencv.org/4.9.0/d7/dfc/group__highgui.html
    
    def create_trackbars(self):
        
        cv2.namedWindow( 'Processed Frame' )
        
        cv2.createTrackbar( 'Canny Thresh Low', 'Processed Frame', 
                            self.canny_threshold_low, 500, self.nothing ) 
        
        cv2.createTrackbar( 'Canny Thresh High', 'Processed Frame',
                            self.canny_threshold_high, 500, self.nothing )
        
        cv2.createTrackbar( 'Min Contour Length', 'Processed Frame',
                            self.min_contour_length, 500, self.nothing )
        
        cv2.createTrackbar( 'Blur Size', 'Processed Frame',
                            self.blur_size, 11, self.nothing )
        
        cv2.createTrackbar( 'Closure Kern Size', 'Processed Frame',
                            self.kernel_size, 11, self.nothing )

    #update values from trackbars       
    def update_trackbar_values( self, processed_frame ):
        
        cv2.imshow( 'Processed Frame', processed_frame )
    
        self.canny_threshold_low = cv2.getTrackbarPos( 'Canny Thresh Low', 
                                                        'Processed Frame' )
            
        self.canny_threshold_high = cv2.getTrackbarPos( 'Canny Thresh High', 
                                                        'Processed Frame' )
            
        self.min_countour_area = cv2.getTrackbarPos( 'Min Contour Length', 
                                                     'Processed Frame' )
            
        #update blur & kernel from trackbar & ensure it's always odd & >0
        new_blur_size = cv2.getTrackbarPos( 'Blur Size', 'Processed Frame' )
            
        self.blur_size = self.make_odd( new_blur_size) 

        new_kernel_size = cv2.getTrackbarPos( 'Kernel Size', 
                                             'Processed Frame' )
            
        self.kernel = self.make_odd( new_kernel_size ) 
        
    #empty callback function for trackbars    
    def nothing( self, x ):
        
        pass
    
    