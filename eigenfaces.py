import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

from matplotlib.pyplot import imread
from numpy import linalg

#-------------------------------------------------------------------

class 	eigenfaces:
	def 	rgb_to_grey(rgb):
		r, g, b, e = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2], rgb[:,:,3]
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#		gray = r + g + b + e
		return gray

	def		mean_face(self):
		i = 0
		image = []
		png = glob.glob("witness_img/*.png")
		while (i < len(png)):
			image.append(self.rgb_to_grey(imread(png[i], True)))
			image[i] = np.reshape(image[i], 270 * 270)
			if (i == 0):
				img = np.copy(image[i])
			else:
				img = np.vstack((img, image[i]))
			i = i + 1
		moyenne = np.mean(img, 0)
#		print("moyenne="+str(moyenne))
#		mpimg.imsave("img.png", moyenne.reshape(270, 270))
		return (img, moyenne, len(png))
	
	def 	process(self):
		img, moyenne, nb_img = self.mean_face(self)
		phi = img - moyenne
		eigenfaces, sigma, v = linalg.svd(phi.transpose(), full_matrices=False)
		for i in range(eigenfaces.shape[1]):
			mpimg.imsave("eigenfaces/img{:d}.png".format(i), eigenfaces[:,i].reshape(270,270))
		weights = np.dot(phi, eigenfaces)
		for j in range(nb_img):
			for i in range(nb_img):
				new = moyenne + np.dot(weights[j, :i], eigenfaces[:, :i].T)
				img_id = str(j)+"_"+str(i)
				mpimg.imsave("img{:d}.{:d}.png".format(i, j), new.reshape(270,270))



#-------------------------------------------------------------------

if __name__ == "__main__":
	eigen = eigenfaces
	eigen.process(eigen)
