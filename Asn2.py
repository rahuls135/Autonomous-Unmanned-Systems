import cv2
import numpy as np

def mul_color_detector():
    cap = cv2.VideoCapture('BarrelVideo.mp4')

    while True:
        _, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Orange color
        low_orange = np.array([10, 130, 20])
        high_orange = np.array([15, 245, 255])
        orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
        orange = cv2.bitwise_and(frame, frame, mask=orange_mask)

        gray = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale image to binary
        ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
 
        # Display the binary image
 
 
        # To detect object contours, we want a black background and a white 
        # foreground, so we invert the image (i.e. 255 - pixel value)
        inverted_binary = ~binary
        # cv2.imshow('Inverted binary image', inverted_binary)
 
        # Find the contours on the inverted binary image, and store them in a list
        # Contours are drawn around white blobs.
        # hierarchy variable contains info on the relationship between the contours
        contours, hierarchy = cv2.findContours(inverted_binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
        
        with_contours = cv2.drawContours(frame, contours, -1,(255,0,255),3)

        cv2.imshow('All contours with bounding box', with_contours)

        key = cv2.waitKey(1)
        if key == 27:
            break

if __name__ == '__main__':
    # single_color_detector()
    mul_color_detector()
