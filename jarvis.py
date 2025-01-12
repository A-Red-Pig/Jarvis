import numpy as np
import cv2
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
import keyboard
import speech_recognition as sr
import threading
import pythoncom
import signal
import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QRect, QThread, pyqtSignal, QObject, QPoint
from PyQt5.QtGui import QColor, QFont
from pynput.keyboard import Controller, Key
from word2number import w2n
import time
from queue import Queue

# Add global flag for program state
running = True

# Add global variables at the top of the file
cap = None
holistic = None

def signal_handler(sig, frame):
    global running
    print('\nShutting down gracefully...')
    running = False
    sys.exit(0)

def camera_tracking():
    def initialize_audio():
        # Initialize COM and keyboard
        pythoncom.CoInitialize()
        keyboard = Controller()
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume)), keyboard

    def initialize_camera_and_mediapipe():
        global cap, holistic  # Make these global so they can be accessed by voice control
        cap = cv2.VideoCapture(0)
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            refine_face_landmarks=True,
            smooth_landmarks=True
        )
        mp_draw = mp.solutions.drawing_utils
        return cap, mp_holistic, holistic, mp_draw

    def main():
        global running, cap, holistic
        # Initialize audio control and keyboard
        volume, keyboard = initialize_audio()
        
        # Initialize camera and MediaPipe
        cap, mp_holistic, holistic, mp_draw = initialize_camera_and_mediapipe()

        # Create a named window and button
        cv2.namedWindow('Webcam Tracking')
        def quit_button(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if 10 <= x <= 110 and 10 <= y <= 40:  # Button area
                    cv2.destroyAllWindows()
                    cap.release()
                    holistic.close()
                    os._exit(0)  # Force exit all threads
        
        cv2.setMouseCallback('Webcam Tracking', quit_button)

        # Custom drawing specs
        pose_drawing_spec = mp_draw.DrawingSpec(
            color=(0, 255, 0),
            thickness=2,
            circle_radius=2
        )

        hand_drawing_spec = mp_draw.DrawingSpec(
            color=(255, 0, 0),
            thickness=2,
            circle_radius=2
        )

        face_drawing_spec = mp_draw.DrawingSpec(
            color=(0, 0, 255),
            thickness=1,
            circle_radius=1
        )

        # Volume control variables
        current_volume = None  # Initialize as None to indicate no volume set yet
        last_y = None
        pinch_start_time = None
        is_pinching = False
        last_volume_change_time = 0

        try:
            while True:
                # Read frame from webcam
                ret, frame = cap.read()
                if not ret:
                    continue

                # Draw quit button
                cv2.rectangle(frame, (10, 10), (110, 40), (0, 0, 255), -1)
                cv2.putText(frame, 'Quit', (35, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame with Holistic
                results = holistic.process(frame_rgb)

                # Get image dimensions
                height, width = frame.shape[:2]

                # Handle volume control with hand gestures
                if results.right_hand_landmarks:
                    landmarks = results.right_hand_landmarks.landmark
                    
                    # Get thumb and index positions
                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]
                    middle_tip = landmarks[12]
                    ring_tip = landmarks[16]
                    pinky_tip = landmarks[20]
                    wrist = landmarks[0]
                    
                    # Calculate hand size (distance from wrist to middle finger base)
                    hand_size = math.sqrt(
                        (landmarks[9].x - wrist.x)**2 + 
                        (landmarks[9].y - wrist.y)**2
                    )
                    
                    # Calculate distance between thumb and index
                    pinch_distance = math.sqrt(
                        (thumb_tip.x - index_tip.x)**2 + 
                        (thumb_tip.y - index_tip.y)**2
                    )
                    
                    # Calculate relative pinch distance as a ratio of hand size
                    relative_pinch_distance = pinch_distance / hand_size
                    
                    # Check if other fingers are far enough
                    other_fingers_away = all([
                        math.sqrt((f.x - thumb_tip.x)**2 + (f.y - thumb_tip.y)**2) > pinch_distance * 2
                        for f in [middle_tip, ring_tip, pinky_tip]
                    ])
                    
                    # Minimum hand size threshold (adjust as needed)
                    if hand_size > 0.1 and other_fingers_away:
                        if relative_pinch_distance < 0.2:  # Relative pinching threshold
                            current_time = cv2.getTickCount() / cv2.getTickFrequency()
                            
                            if not is_pinching:
                                # Get current system volume when pinch starts
                                try:
                                    current_volume = int(volume.GetMasterVolumeLevelScalar() * 100)
                                    print(f"Starting volume control at {current_volume}%")
                                except Exception as e:
                                    print(f"Error getting current volume: {e}")
                                    current_volume = 50  # fallback value
                                
                                pinch_start_time = current_time
                                last_y = thumb_tip.y
                                is_pinching = True
                            elif last_y is not None:
                                # Calculate movement relative to hand size
                                movement = (last_y - thumb_tip.y) / hand_size
                                
                                # Check if movement isn't too fast
                                if abs(movement) < 0.5:  # Adjust threshold as needed
                                    # Update volume
                                    volume_change = int(movement * 100)
                                    if abs(volume_change) >= 2:  # Minimum change threshold
                                        new_volume = current_volume + (2 if volume_change > 0 else -2)
                                        new_volume = max(0, min(100, new_volume))
                                        
                                        if new_volume != current_volume:
                                            current_volume = new_volume
                                            volume.SetMasterVolumeLevelScalar(current_volume / 100, None)
                                            
                                            # Simulate volume key press to show Windows volume HUD
                                            if volume_change > 0:
                                                keyboard.press(Key.media_volume_up)
                                                keyboard.release(Key.media_volume_up)
                                            else:
                                                keyboard.press(Key.media_volume_down)
                                                keyboard.release(Key.media_volume_down)
                                        
                                last_y = thumb_tip.y
                        else:
                            is_pinching = False
                            last_y = None

                # Draw pose landmarks if detected
                if results.pose_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        pose_drawing_spec,
                        pose_drawing_spec
                    )

                # Draw hand landmarks if detected
                for hand_landmarks, hand_connections in [
                    (results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS),
                    (results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                ]:
                    if hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            hand_connections,
                            hand_drawing_spec,
                            hand_drawing_spec
                        )

                # Draw face mesh if detected
                if results.face_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        face_drawing_spec,
                        face_drawing_spec
                    )

                # Show frame
                cv2.imshow('Webcam Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            holistic.close()

    # if __name__ == "__main__":
    main()

def voice_control():
    global running
    
    # Initialize COM
    pythoncom.CoInitialize()
    
    # Initialize audio control
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    
    # Initialize speech recognition with adjusted settings
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.0
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 1.0
    
    # Define wake words that sound like "jarvis"
    WAKE_WORDS = ["jarvis", "target", "character", "carpet", "purpose", 
                  "sharpest", "darkness", "traffic", "surface", "nervous", 
                  "travis", "jarves", "jarbus", "tardis"]
    
    while running:
        try:
            with sr.Microphone() as source:
                print("Listening...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                recognizer.dynamic_energy_threshold = True
                recognizer.energy_threshold = 4000
                
                audio = recognizer.listen(source, 
                                        timeout=None,
                                        phrase_time_limit=None)
                
                if not running:
                    break
                
                text = recognizer.recognize_google(audio).lower()
                print(f"Recognized: {text}")
                
                # Add reboot/restart command
                if any(word in text for word in WAKE_WORDS) and any(word in text for word in ["reboot", "restart"]):
                    print("Rebooting Jarvis...")
                    script_path = os.path.abspath(sys.argv[0])
                    os.execv(sys.executable, ['python'] + [script_path])
                
                # Add quit command - force terminate process
                if any(word in text for word in WAKE_WORDS) and any(word in text for word in ["quit", "exit", "stop", "shutdown", "shut down", "close"]):
                    print("Shutting down Jarvis...")
                    import subprocess
                    subprocess.run(['taskkill', '/F', '/PID', str(os.getpid())], capture_output=True)
                    os._exit(0)

                # Add volume control commands
                # Check for "set volume to X" pattern
                if "set" in text and "volume" in text:
                    try:
                        # Extract all text after "to" and find all numbers
                        volume_str = text.split("to")[-1].strip()
                        # Split into words and convert each to number if possible
                        words = volume_str.split()
                        numbers = []
                        for word in words:
                            try:
                                num = w2n.word_to_num(word)
                                numbers.append(num)
                            except:
                                continue
                        
                        # Use the last number found if any exist
                        if numbers:
                            volume_level = numbers[-1]
                            # Ensure volume is between 0 and 100
                            volume_level = max(0, min(100, volume_level))
                            # Set the volume
                            volume.SetMasterVolumeLevelScalar(volume_level / 100, None)
                            print(f"Volume set to {volume_level}%")
                            update_volume_with_notification(volume_level)
                        else:
                            print("No valid volume level found")
                    except Exception as e:
                        print(f"Error setting volume: {e}")
                

                if "what" in text and "time" in text and "is" in text:
                    current_time = time.strftime("%I:%M %p")
                    message = f"It is {current_time}"
                    # popup_manager.notification_queue.put(message)
                    custom_notification(message)

        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        except Exception as e:
            print(f"Error in voice control loop: {e}")
            if not running:
                break

class NotificationThread(QThread):
    show_notification = pyqtSignal(str)

    def __init__(self, message):
        super().__init__()
        self.message = message

    def run(self):
        self.show_notification.emit(self.message)
        self.msleep(100)

class PopupManager(QObject):
    def __init__(self):
        super().__init__()
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.active_popups = []
        self.notification_queue = Queue()
        self.current_popup = None
        
        # Create timer to process queue
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_queue)
        self.timer.start(100)

    def process_queue(self):
        if not self.notification_queue.empty() and (self.current_popup is None or not self.current_popup.isVisible()):
            message = self.notification_queue.get()
            self.show_notification(message)

    def show_notification(self, message):
        try:
            # Create new popup
            self.current_popup = PopupWindow(message)
            self.current_popup.closed.connect(self.on_popup_closed)
            self.current_popup.show()
        except Exception as e:
            print(f"Error in show_notification: {e}")

    def on_popup_closed(self):
        self.current_popup = None

    def queue_notification(self, message):
        self.notification_queue.put(message)

class PopupWindow(QMainWindow):
    closed = pyqtSignal()

    def __init__(self, message):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Create labels with word wrap enabled
        jarvis_label = QLabel("JARVIS")
        jarvis_label.setStyleSheet("color: #00BFFF; font-weight: bold; font-size: 10pt")
        
        message_label = QLabel(message)
        message_label.setStyleSheet("color: white; font-size: 11pt")
        message_label.setWordWrap(True)  # Enable word wrap
        message_label.adjustSize()  # Adjust size to content
        
        # Add labels to layout
        layout.addWidget(jarvis_label, alignment=Qt.AlignCenter)
        layout.addWidget(message_label, alignment=Qt.AlignCenter)
        
        # Set window style
        central_widget.setStyleSheet("background-color: rgba(30, 30, 30, 0.9); border-radius: 5px")
        
        # Calculate required width and height
        message_width = message_label.fontMetrics().boundingRect(message).width()
        width = min(max(250, message_width + 40), 400)  # Min 250px, Max 400px
        
        # Set size and position
        self.setFixedWidth(width)
        self.adjustSize()  # Adjust height automatically
        height = self.height()
        
        screen = QApplication.primaryScreen().geometry()
        self.move(-width, 20)
        
        # Slide in animation
        self.anim = QPropertyAnimation(self, b"geometry")
        self.anim.setDuration(300)
        self.anim.setStartValue(QRect(-width, 20, width, height))
        self.anim.setEndValue(QRect(20, 20, width, height))
        self.anim.start()
        
        # Setup fade out timer
        QTimer.singleShot(5000, self.fade_out)
    
    def fade_out(self):
        self.fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self.fade_anim.setDuration(1000)
        self.fade_anim.setStartValue(1.0)
        self.fade_anim.setEndValue(0.0)
        self.fade_anim.finished.connect(self.on_fade_finished)
        self.fade_anim.start()
    
    def on_fade_finished(self):
        self.close()
        self.closed.emit()
        self.deleteLater()

def update_volume_with_notification(volume_level):
    try:
        message = f"Volume set to {volume_level}%"
        global popup_manager
        if popup_manager is not None:
            popup_manager.queue_notification(message)
    except Exception as e:
        print(f"Error showing volume notification: {e}")

def custom_notification(message):
    try:
        # message = f"Volume set to {volume_level}%"
        global popup_manager
        if popup_manager is not None:
            popup_manager.queue_notification(message)
    except Exception as e:
        print(f"Error showing volume notification: {e}")

if __name__ == "__main__":
    # Add signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Create QApplication instance at startup
        app = QApplication(sys.argv)
        
        # Create global popup manager in main thread
        popup_manager = PopupManager()
        
        # Create threads for each function
        camera_thread = threading.Thread(target=camera_tracking)
        voice_thread = threading.Thread(target=voice_control)
        
        # Start both threads
        camera_thread.start()
        voice_thread.start()
        
        # Start Qt event loop
        app.exec_()
        
        # After Qt event loop ends, clean up
        running = False
        camera_thread.join()
        voice_thread.join()
        sys.exit(0)
        
    except KeyboardInterrupt:
        print('\nShutting down gracefully...')
        running = False
        sys.exit(0)
    except Exception as e:
        print(f"Error in main: {e}")
        running = False
        sys.exit(1)