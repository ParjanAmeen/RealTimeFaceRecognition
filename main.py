# This program uses a webcam and recognizes faces. It will compare
# the face on the webcam to the jpg file stored in the directory. If the faces
# are the same a name tag will appear under the face. If they are not similar an
# "unknown" prompt will come up. Upload a picture of your self into the directory where
# my face is stored and follow the comment code to make some changes to have the program
# scan for your face

# Importing the packets for face recognition
import numpy as np
import face_recognition as fr
import cv2

# Video capture through webcam enabled
video_capture = cv2.VideoCapture(0)

# Change the variable names to your name and upload a picture of your face
# Loads the image in the directory and encodes it. Enter the picture file of you in the quotes
parjan_image = fr.load_image_file('Parjan.jpg')
parjan_face_encoding = fr.face_encodings(parjan_image)[0]

known_faces_encodings = [parjan_face_encoding]

# If I had more names we would just pass more in this array. Pass in your name
known_face_names = ['Parjan']

while True:
    ret, frame = video_capture.read()

    # Calling the frame function
    rgb_frame = frame[:, :, ::-1]

    # Catches the call for face locations and face encodings
    face_location = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_location)

    # Start of for loop
    for (top, right, bottom, left), face_encodings in zip(face_location, face_encodings):

        # Set the matches variable to catch the compare faces function
        matches = fr.compare_faces(known_faces_encodings, face_encodings)

        # If the face is not recognized then label with 'Unknown'
        name = 'Unknown'

        # Catching the call for face distance and pacing in the encoded images
        face_distance = fr.face_distance(known_faces_encodings, face_encodings)

        # Set the best match index to catch the argmin function
        best_match_index = np.argmin(face_distance)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Creating the rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)

        # Setting the font and calling the put text function to put some text under the rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # End of for loop
    # Call the image show function
    cv2.imshow('Webcam_facerecognition', frame)

    # Allow the user to quit the webcam by pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Break from the webcam
video_capture.release()
cv2.destroyAllWindows()