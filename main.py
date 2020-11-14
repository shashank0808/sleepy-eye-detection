from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound
frequency = 2500
duration = 1000

def eyeAspectRatio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

count = 0
Thresh = 0.25 
ear_Frames = 20 
Predict = 'shape_predictor_68_face_landmarks.dat'

webcam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(Predict)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
while True:
	ret, frame = webcam.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 0)

	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eyeAspectRatio(leftEye)
		rightEAR = eyeAspectRatio(rightEye)

		ear = (leftEAR + rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0),1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0),1)
		if ear <Thresh:
			count = count + 1
			#print(count)
			if count >= ear_Frames:
				cv2.putText(frame, 'DROWSINESS DETECTED', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
				winsound.Beep(frequency, duration)
		else:
			count = 0

	cv2.imshow('Frame', frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		print('quit')
		break
webcam.release()
cv2.destroyAllWindows()