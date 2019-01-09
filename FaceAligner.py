import dlib
import cv2
import os
import glob

print('Using dlib version: {}'.format(dlib.__version__))
print('Using cv2 version: {}'.format(cv2.__version__))

def FaceAligner(face_file_path, output_path, predictor_path='./shape_predictor_5_face_landmarks.dat'):
    # Load all the models we need: a detector to find the faces, a shape predictor
    # to find face landmarks so we can precisely localize the face
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    
    img=cv2.imread(face_file_path)
    if img is None:
        return
    
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    num_faces = len(dets)

    if num_faces == 0:
        return
    
    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img, detection))

    # Save Image
    image = dlib.get_face_chip(img, faces[0])
    dlib.save_image(image, output_path)

root="./Images/*" # The directory where the downloaded images are housed
dst_dir="./Aligned" # The directory to place the cropped and resized images
os.mkdir(dst_dir)
src_dir=glob.glob(root)

# Will crop and resize the downloaded images using OpenCV and place the results in the destination directory declared above
for path in src_dir:
    dst = os.path.join(dst_dir, path.split('/')[2])
    os.mkdir(dst)
    for img in os.listdir(path):
        FaceAligner(os.path.join(path, img), os.path.join(dst, img))