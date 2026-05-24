import cv2
import time
from .motion_detector import MotionDetector
from .classifier import WildlifeClassifier

class BioSpikePipeline:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        self.detector = MotionDetector()
        self.classifier = WildlifeClassifier()

    def run(self):
        print("[INFO] BioSpike-Vision Edge Pipeline Active. Monitoring...")
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # STEP 1: Low-power check
                if self.detector.has_motion(frame):
                    # STEP 2: High-power classification triggered only on event
                    start_time = time.time()
                    animal_type = self.classifier.predict(frame)
                    latency = (time.time() - start_time) * 1000
                    
                    # Log event metrics
                    print(f"[TRIGGER] Motion Detected! Classification: {animal_type} | Latency: {latency:.2f}ms")
                    
                    # UI Overlay (Visual representation for demonstrations)
                    cv2.putText(frame, f"ALERT: {animal_type}", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # System mimics a low-power deep sleep state when idle
                    cv2.putText(frame, "System Mode: Low-Power Standby", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("BioSpike-Vision Edge Feed", frame)

                # Break loop with 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Pipeline shut down gracefully.")