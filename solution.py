"""
Derin Mavi Line Follower Challenge

Bu dosyayı düzenleyerek kendi çizgi izleme algoritmanızı geliştirin!
Aşağıdaki solution() fonksiyonunu tamamlayın.

Başarılar!
"""

import cv2
import numpy as np


def solution(image, current_speed, current_steering):
    """  
    Args:
        image: Robotun kamerasından gelen 64x64 pixel BGR görüntü (numpy array)
               
        current_speed: Robotun mevcut hızı (float)
                      
        current_steering: Robotun mevcut direksiyon açısı (float, -1 ile 1 arası)
                         - -1: Tam sol
                         -  0: Düz
                         -  1: Tam sağ
    
    Returns:
        target_speed: Robotun hedef hızı (float)

        steering: Robotun hedef direksiyon açısı (float, -1 ile 1 arası)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    roi = gray[20:50, :]
    _, cols = np.where(roi < 60) # x-values are enough.
    if len(cols) == 0:
        return 2.0, current_steering
    track_center = np.mean(cols) # center of the road.
    image_center = 32.0 # center of the camera image.
    error = track_center - image_center

    steering = error / 20.0 # image_center:32, track_center:48-->16/20=0.8
    # We can increase or decrease the denominator for calmer or more aggressive turns, respectively.
    # If the denominator is choosen as 32.0 (in order not to write steering = np.clip(steering, -1.0, 1.0)), the car turns very late.
    # If the denominator is choosen as 1.0 (very small example), the car oscillates.
    steering = np.clip(steering, -1.0, 1.0) # for the values <-1 or >1 (sharp turn)
    # If the track is straight, go at maximum speed. If approaching a curve, slow down. 
    base_speed = 23.0 # We can increase this value for more aggresive driving. 
    min_speed = 4.0 # If we increase base_speed, we must decrease this value so that the car can turn the curve.
    # But we need also pay attention to physical limitations. The car may not decrease its speed enough if we choose base speed very big.
    target_speed = base_speed - (abs(steering) * (base_speed - min_speed)) # If steering=0, the target speed does not change.
    # If the steering is -1 or 1 (both will be 1 in abs function), target speed will be the minimum speed.
    # I reached the optimal result at 27.0-15.0.
    return target_speed, steering
