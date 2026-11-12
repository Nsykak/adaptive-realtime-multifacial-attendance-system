import cv2  # OpenCV library for image and video processing
import numpy as np  # For numerical operations
import face_recognition as face_rec  # For facial recognition
import os  # For file and directory operations
import pyttsx3 as textSpeak  # For text-to-speech functionality
from datetime import datetime  # To handle date and time

if __name__ == '__main__':
    # Initialize the text-to-speech engine
    engine = textSpeak.init()
    
    # Function to resize images for faster processing
    def resize(img, size):
        width = int(img.shape[1] * size)  # Calculate new width
        height = int(img.shape[0] * size)  # Calculate new height
        dimension = (width, height)  # New dimensions as a tuple
        return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)  # Resize the image
    
    # Path to the folder containing attendee images
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Attendee_images')
    os.makedirs(path, exist_ok=True)
    attendeeImg = []  # List to store images of attendees
    attendeeName = []  # List to store attendee names
    attendedNames = []  # List to track names of attendees who have been marked
    myList = os.listdir(path)  # List all files in the specified path
    
    # Load attendee images and names
    for cl in myList:
        curimg = cv2.imread(f'{path}/{cl}')  # Read each image file
        if curimg is None:
            continue
        attendeeImg.append(curimg)  # Append image to the list
        attendeeName.append(os.path.splitext(cl)[0])  # Extract and store the name without file extension
    
    # Function to find face encodings
    def findEncoding(images):
        imgEncodings = []
        valid_names = []  # List to store face encodings
        for idx, img in enumerate(images):
            img = resize(img, 0.50)  # Resize image for faster processing
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            encodings = face_rec.face_encodings(img)  # Get face encodings
            if encodings:  # Check if at least one face encoding exists
                valid_names.append(attendeeName[idx])
                imgEncodings.append(encodings[0])  # Append the first encoding
            else:
                print(f"No face detected in image {idx + 1}. Skipping...")  # Log if no face is detected
        attendeeName[:] = valid_names
        return imgEncodings  # Return the list of encodings
    
    # Function to mark attendance
    def MarkAttendance(name):
        import csv
        now = datetime.now()
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attendance.csv')
        rows = []
        if os.path.exists(csv_path):
            with open(csv_path, newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
        if any(row.get('Name') == name and row.get('Date') == now.date().isoformat() for row in rows):
            return
        needs_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            if needs_header:
                writer.writerow(['Name', 'Time', 'Date'])
            writer.writerow([name, now.strftime('%H:%M'), now.date().isoformat()])
        if name not in attendedNames:
            attendedNames.append(name)
        engine.say(f'Welcome to our event, {name}')
        engine.runAndWait()
    
    # Encode the faces of attendees
    EncodeList = findEncoding(attendeeImg)
    
    # Start video capture
    vid = cv2.VideoCapture(0)  # Open the default webcam
    
    # Set the resolution of the webcam capture (increase width and height)
    if not vid.isOpened():
        raise RuntimeError('Cannot open webcam. Check camera permissions and device index.')
    
    vid.set(3, 1280)  # Set width to 1280 (you can adjust this to any resolution)
    vid.set(4, 720)   # Set height to 720 (you can adjust this to any resolution)
    # Set a crowd threshold
    CROWD_THRESHOLD = 5  # Maximum allowed faces before triggering an alert
    
    try:
        while True:
            success, frame = vid.read()  # Capture a frame from the webcam
            if not success:
                break  # Exit loop if the frame capture fails
        
            clean_frame = frame.copy()
        
            # Resize frame for faster processing
            Smaller_frames = cv2.resize(frame, None, fx=0.50, fy=0.50)  # Downscale frame
            rgb_small_frame = cv2.cvtColor(Smaller_frames, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
            # Detect faces in the frame
            facesInFrame = face_rec.face_locations(rgb_small_frame, model="hog")  # Get face locations
        
            # Monitor crowd levels
            number_of_faces = len(facesInFrame)  # Count the number of faces detected
            if number_of_faces > CROWD_THRESHOLD:
                print(f"Alert: Overcrowding detected! {number_of_faces} faces found.")  # Log overcrowding alert
                engine.say(f"Alert: Overcrowding detected! {number_of_faces} faces found.")  # Speak alert
                engine.runAndWait()
        
            # Encode multiple faces in the frame
            encodeFacesInFrame = face_rec.face_encodings(rgb_small_frame, facesInFrame)
        
            for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame):
                matches = face_rec.compare_faces(EncodeList, encodeFace)  # Check for matches
                facedis = face_rec.face_distance(EncodeList, encodeFace)  # Calculate face distances
                matchIndex = np.argmin(facedis) if len(facedis) > 0 else -1  # Get the closest match index
        
                # Improved face recognition logic
                if matches and matchIndex != -1 and facedis[matchIndex] < 0.5:  # Adjusted threshold
                    name = attendeeName[matchIndex].upper()  # Get attendee name
                else:
                    name = "UNKNOWN"  # Mark as unknown if no match is found
        
                # Scale back face location to original frame size
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
        
                # Optionally expand the bounding box to capture more features (e.g., hair)
                expansion = 30
                y1 = max(y1 - expansion, 0)
                x1 = max(x1 - expansion, 0)
                y2 = min(y2 + expansion, frame.shape[0])
                x2 = min(x2 + expansion, frame.shape[1])
        
                color = (0, 165, 255) if name != "UNKNOWN" else (0, 0, 255)  # Set color for bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Draw bounding box
                cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), color, cv2.FILLED)  # Draw name tag background
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)  # Display name
        
                if name == "UNKNOWN":
                    snapshot = clean_frame[y1:y2, x1:x2]  # Capture snapshot of unknown face
                    snapshot_resized = cv2.resize(snapshot, (300, 300))  # Resize snapshot for display
                    cv2.imshow("Unknown Face", snapshot_resized)  # Show unknown face
        
                    key = cv2.waitKey(1) & 0xFF  # Wait for key input
                    if key == ord('s'):  # If 's' is pressed, save the face
                        label = input("Enter the name for the new face: ").strip()  # Get name input
        
                        if label and all(c.isalnum() or c in ' _-' for c in label):  # Save new face if a name is provided
                            new_image_path = os.path.join(path, f"{label}.jpg")  # Generate file path
                            if os.path.exists(new_image_path):
                                print('Name already exists; choose another label.')
                                continue
                            if not cv2.imwrite(new_image_path, snapshot_resized):
                                print('Could not save image.')
                                continue  # Save the image
        
                            # Read and encode the saved image
                            new_img = cv2.imread(new_image_path)
                            new_img = resize(new_img, 0.50)  # Resize image
                            new_img_rgb = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                            new_encodings = face_rec.face_encodings(new_img_rgb)  # Encode new face
        
                            if new_encodings:  # Check if encoding exists
                                new_encoding = new_encodings[0]
                                attendeeImg.append(new_img)  # Add to attendee images
                                attendeeName.append(label.upper())  # Add name
                                EncodeList.append(new_encoding)  # Add encoding
                                print(f"New face for {label} added successfully!")
                            else:
                                os.remove(new_image_path)
                                print("No face detected in the image. Please try again.")
                        else:
                            print("Name not provided. Face not saved.")
        
                else:
                    MarkAttendance(name)  # Mark attendance for recognized faces
        
            # Display the video feed
            cv2.imshow('video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop if 'q' is pressed
                break
        
        # Generate acknowledgment speech for attendees
        if attendedNames:
            acknowledgment = "Thank you to all for coming to Adaptive realtime multi-facial attendance system. Here are the names of the people who attended: "
            acknowledgment += ", ".join(attendedNames)
            engine.say(acknowledgment)  # Speak acknowledgment
            engine.runAndWait()
        
    finally:
        vid.release()
        cv2.destroyAllWindows()
