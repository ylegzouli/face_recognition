import cv2
import pickle
#import common as c
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.pyplot import imread

#-------------------------------------------------------------------

class 	Tester:
	def		affichage(self, face, name):
		if name == "Inconnu":
			tmp = np.array(imread("witness_img/inconnu.png", cv2.IMREAD_GRAYSCALE))
			img = cv2.resize(tmp, dsize=(270, 270), interpolation=cv2.INTER_CUBIC)
		else:
			img = np.array(imread("witness_img/{:s}/{:s}1.png".format(name, name), cv2.IMREAD_GRAYSCALE))
			
		plt.figure()
		plt.subplot(2, 1, 1)
		plt.imshow(face)
		plt.subplot(2, 1, 2)
		plt.imshow(img)
		plt.show()

	def		tester(self):
		face = np.array(imread("save_img/img.png", cv2.IMREAD_GRAYSCALE))
		res = cv2.resize(face, dsize=(270, 270), interpolation=cv2.INTER_CUBIC)
		recognizer = cv2.face.LBPHFaceRecognizer_create()
		recognizer.read("files/Id_base.yml")
		with open("files/Id_base", "rb") as f:
			og_labels = pickle.load(f)
			labels = {v:k for k, v in og_labels.items()}
		id_, conf=recognizer.predict(res)
		print(conf)
		if conf < 45:
			percent = (200 - conf) / 2
			name = labels[id_]
		else:
			name = "Inconnu"
			percent = 0
		print("Name: {:s}".format(name))
		print("{:d}%".format(int(percent)))
		self.affichage(self, face, name)

#-------------------------------------------------------------------

if __name__ == "__main__":
	test = Tester
	test.tester(test)
