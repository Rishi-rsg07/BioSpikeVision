import cv2

class MotionDetector:
    def __init__(self, threshold=25, min_area=500):
        """
        Low-power motion gating using frame differencing.
        threshold: Minimum pixel intensity difference to consider as change.
        min_area: Minimum contour area (in pixels) to trigger the classifier.
        """
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame = None

    def has_motion(self, frame):
        # Converting to grayscale and blur to remove high-frequency noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False

        # Computing absolute difference between current and previous frame
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Finding contours of moving objects
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Updating background memory frame
        self.prev_frame = gray

        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                return True  # Motion detected, wake up the AI
                
        return False