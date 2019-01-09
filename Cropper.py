import os
import glob
import cv2

### Recognize, crop, and resize with OpenCV
root="./Images/*" # The directory where the downloaded images are housed
dst_dir="./Cropped" # The directory to place the cropped and resized images
os.mkdir(dst_dir)
src_dir=glob.glob(root)

# Will crop and resize the downloaded images using OpenCV and place the results in the destination directory declared above
for path in src_dir:
    dst = os.path.join(dst_dir, path.split('/')[2])
    os.mkdir(dst)
    for img in os.listdir(path):
        image = cv2.imread(os.path.join(path, img))
        if image is None:
            continue
        image_grey=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert image to greyscale
        cascade=cv2.CascadeClassifier("/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml") #Import Classifier
        face_list=face_list=cascade.detectMultiScale(image_grey, scaleFactor=1.1, minNeighbors=2,minSize=(64,64)) # Face recognition
        # If faces were detected
        if len(face_list) > 0: 
            for rect in face_list:
                x,y,width,height=rect
                image=image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                if image.shape[0] < 64: 
                    continue
                image = cv2.resize(image,(64,64))
            fileName=os.path.join(dst, img)
            cv2.imwrite(fileName, image) # Save image
        else:
            continue