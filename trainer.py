import numpy as np
#import common as c
import pickle
import cv2
import os

from get_face import get_img

#----------------------------------------------------------------

class 	Trainer:	
	def 	get_file(label, cur_id, x_train, y_train, dir_face):
		for root, dirs, files in os.walk(dir_face):
			if len(files):
				name = root.split("/")[-1]
				for img in files:
					if img.endswith("png"):
						path = os.path.join(root, img)
						if not name in label:
							label[name] = cur_id
							cur_id += 1
						y = label[name]
						x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#						x = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (c.min_size, c.min_size))
#						fm = cv2.Laplacian(x, cv2.CV_64F).var()
#						if fm > 250:
						x_train.append(x)
						y_train.append(y)

	def 	process(self):
		temp = str(input("New dataBase ? (y/n) "))
		if (temp == 'y'):
			data = get_img
			data.process(data)
		dir_face = "witness_img/"
		cur_id = 0
		label = {}
		x_train = []
		y_train = []
		self.get_file(label, cur_id, x_train, y_train, dir_face)
		with open("files/Id_Base", "wb") as f:
			pickle.dump(label, f)
		x_train = np.array(x_train)
		y_train = np.array(y_train)
		recognizer = cv2.face.LBPHFaceRecognizer_create()
		recognizer.train(x_train, y_train)
		recognizer.save("files/Id_base.yml")
		print("Train [OK]")


#----------------------------------------------------------------

if __name__ == "__main__":
	train = Trainer
	train.process(train)
