from random import randint
import cv2
import sys
import os
import numpy as np
      
CASCADE="C:/Users/shintani/Desktop/Study/Tensorflow/FaceBoss/haarcascade_frontalface_alt.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

def detect_faces(path_image_in, folder_out, image_name):
	name = image_name[:-4]
	extention = image_name[-4:]

	image_=cv2.imread(path_image_in)
	# print (type(image_))

	if isinstance(image_, np.ndarray):
		image_grey=cv2.cvtColor(image_,cv2.COLOR_BGR2GRAY)

		faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)
		print (faces)
		i = 0
		for x,y,w,h in faces:
			sub_img = image_[y-10:y+h+10,x-10:x+w+10]
			image_name_out = os.path.join(folder_out, name + '_' + str(i) + extention)
			cv2.imwrite(image_name_out, sub_img)
			i += 1

def get_face_images(folder_resize, folder_out):
	print ("================ Start Reize ================")
	list_file_in = next(os.walk(folder_resize))[2]
	if len(list_file_in) > 0:

		# Create path out
		# os.chdir(path_folder_out)
		if not os.path.exists(folder_out):
			os.makedirs(folder_out)

		for image_name in list_file_in:
			print (".")
			path_image_in = os.path.join(folder_resize, image_name)

			# Detect and save face
			if os.path.exists(path_image_in):
				detect_faces(path_image_in, folder_out, image_name)
	print ("================ End Reize ================\n")
			

def resize_a_image(img_path_in, img_path_out):

	from PIL import Image
	from keras.preprocessing import image

	scale = 0.5
	im = Image.open(img_path_in)
	img_array = image.img_to_array(im)


	extention = img_path_out[-4:]
	img_path_out = img_path_out.replace(" ", "")
	img_path_out = img_path_out.replace("%", "")
	img_path_out = img_path_out.replace(".", "")
	img_path_out = img_path_out.replace("x", "")
	img_path_out = img_path_out.replace("-", "")
	img_path_out += extention

	# Check before resize
	# print (img_array.shape)
	if img_array.shape[0] > 2000 or img_array.shape[1] > 2000:

		x = int(img_array.shape[0]  * scale)
		y = int(img_array.shape[1] * scale)
		size = (y, x)

		im_resized = im.resize(size, Image.ANTIALIAS)
		img_array = image.img_to_array(im_resized)
		# print (img_array.shape)
		try:
			im_resized.save(img_path_out)
		except:
			print (img_path_out)
	# Copy
	else:
		print ("========================")
		im = Image.open(img_path_in)
		try:
			im.save(img_path_out)
		except:
			print (img_path_out)
def resize_images(folder_in, folder_resize):
	print ("================ Start Reize ================")
	list_file_in = next(os.walk(folder_in))[2]
	if len(list_file_in) > 0:

		# Create path out
		# os.chdir(path_folder_out)
		if not os.path.exists(folder_resize):
			os.makedirs(folder_resize)

		for image_name in list_file_in:
			print (".")
			path_image_in = os.path.join(folder_in, image_name)
			path_image_out = os.path.join(folder_resize, image_name)

			# Detect and save face
			# print (path_image_in)
			if os.path.exists(path_image_in):
				# print ("=================")
				resize_a_image(path_image_in, path_image_out)
	print ("================ End Reize ================\n")


if __name__ == '__main__':
	from sys import argv

	script, folder_in = argv

	# folder_in = 'C:/Users/shintani/Desktop/Study/Tensorflow/FaceBoss/image/source'
	# folder_in = 'C:/Users/shintani/Desktop/Study/Tensorflow/FaceBoss/image/google_image/downloads/nguoi noi tieng'
	folder_in = 'C:/Users/shintani/Desktop/Study/Tensorflow/FaceBoss/image/source/21'

	folder_name = folder_in.split('/')[-1]
	path_parent_folder = folder_in.replace('/' + folder_name, '')

	folder_out =  os.path.join(path_parent_folder, folder_name + '_faces')
	folder_resize =  os.path.join(path_parent_folder, folder_name + '_resize')

	print (folder_in)
	print (folder_resize)
	print (folder_out)
	
	# resize_images(folder_in, folder_resize)
	get_face_images(folder_resize, folder_out)