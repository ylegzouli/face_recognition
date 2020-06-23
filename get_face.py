import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tester import Eigenfaces

#-------------------------------------------------------------------

class	get_img:
	def		save_img(x, y, w, h, frame, n, eigen):
		img = frame[y+2:y+h-1, x+2:x+w-1]
		res = cv2.resize(img, dsize=(270, 270), interpolation=cv2.INTER_CUBIC)
		res2 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
		cv2.imwrite("save_img/img.png", res2)
		print("img{:d}.png: [SAVE]".format(n))
	
	def		process(self, eigen):
		face_cascade2=cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_alt2.xml")
		cap=cv2.VideoCapture(0)
		nb=1
		exit = 1
		while (exit):
			ret, frame=cap.read()
			gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			face=face_cascade2.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
			for x, y, w, h in face:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
				cv2.putText(frame, "face: {:d}".format(nb), (x, y+h+20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
				if cv2.waitKey(1)&0xFF==ord('s'):
					self.save_img(x, y, w, h, frame, nb, eigen)
					exit = 0
				nb=nb+1
#			if cv2.waitKey(1)&0xFF==ord('q'):
#				break
			cv2.imshow('vew', frame)
		cap.release()
		cv2.destroyAllWindows()
		eigen.tester(eigen)

#-------------------------------------------------------------------

if __name__ == "__main__":
	img = get_img
	eigen = Eigenfaces
	img.process(img, eigen)

