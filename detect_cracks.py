#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 23:54:37 2024

@author: sepehr
"""

from Custom_Pkg.Detector import VideoProcessor, \
    MLDetector, CannyDetector, CannyRealTime
    
import time

def main():
    
    #defining all required paths
    
    video_path = 'video_input/video.mp4' 
    model_path = 'runs/segment/train/weights/best.pt'
    output_path_ML = 'video_output/output_ML.mp4'
    output_path_Canny = 'video_output/output_canny.mp4'
    
    #create instance of Canny detector
        
    canny_real_time = CannyRealTime( video_path )
    
    #interactive tuning
    
    canny_real_time.play_video( half_speed = False )
    
    time.sleep(2)
        
    #process the video with Canny detector and determined parameters
            
    canny_process = CannyDetector( video_path , 
                                   canny_threshold_low = 75,
                                   canny_threshold_high = 175, 
                                   min_contour_length= 275,
                                   blur_size = 7, 
                                   kernel_size = 3 )
    
    canny_process.process_video( output_path_Canny )
    
    time.sleep(2)

    #create instance of ML detector
    
    ml_detector = MLDetector( video_path, model_path )
    
    #process the video with ML detector
    
    ml_detector.process_video( output_path_ML )
    
    time.sleep(2)
        
    #let's view how well YOLO detected cracks
        
    ml_view = VideoProcessor( output_path_ML )
    
    ml_view.play_video( half_speed = False )
    
    time.sleep(2)

    print( "Finished main script." )

if __name__ == "__main__":
    main()
    