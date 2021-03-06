import glob, os, cv2, math, shutil
from scipy import ndimage

# Path to the directory that will hold the data
base_dir='./Input_Data_Rotate'
os.mkdir(base_dir)

# The directories to house the training images, validation images, and test images
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Creating the directory to house the training images
train_erika_dir = os.path.join(train_dir, 'erika')
os.mkdir(train_erika_dir)
train_asuka_dir = os.path.join(train_dir, 'asuka')
os.mkdir(train_asuka_dir)
train_mai_dir = os.path.join(train_dir, 'mai')
os.mkdir(train_mai_dir)
train_nanase_dir = os.path.join(train_dir, 'nanase')
os.mkdir(train_nanase_dir)
train_nanami_dir = os.path.join(train_dir, 'nanami')
os.mkdir(train_nanami_dir)

# The directory to house the testing images
test_erika_dir = os.path.join(test_dir, 'erika')
os.mkdir(test_erika_dir)
test_asuka_dir = os.path.join(test_dir, 'asuka')
os.mkdir(test_asuka_dir)
test_mai_dir = os.path.join(test_dir, 'mai')
os.mkdir(test_mai_dir)
test_nanase_dir = os.path.join(test_dir, 'nanase')
os.mkdir(test_nanase_dir)
test_nanami_dir = os.path.join(test_dir, 'nanami')
os.mkdir(test_nanami_dir)

fnames = glob.glob("./Aligned/生田絵梨花/*") 
train_len = math.floor(len(fnames) * 0.7)
for i in range(len(fnames)):
    if i < train_len:
        src = fnames[i]
        dst = os.path.join(train_erika_dir, 'erika.{}.jpg'.format(i))
        shutil.copyfile(src, dst)
    else: 
        src = fnames[i]
        dst = os.path.join(test_erika_dir, 'erika.{}.jpg'.format(i))
        shutil.copyfile(src, dst)

fnames = glob.glob("./Aligned/齋藤飛鳥/*")
train_len = math.floor(len(fnames) * 0.7)
for i in range(len(fnames)):
    if i < train_len:
        src = fnames[i]
        dst = os.path.join(train_asuka_dir, 'asuka.{}.jpg'.format(i))
        shutil.copyfile(src, dst)
    else: 
        src = fnames[i]
        dst = os.path.join(test_asuka_dir, 'asuka.{}.jpg'.format(i))
        shutil.copyfile(src, dst)

fnames = glob.glob("./Aligned/白石麻衣/*")
train_len = math.floor(len(fnames) * 0.7)
for i in range(len(fnames)):
    if i < train_len:
        src = fnames[i]
        dst = os.path.join(train_mai_dir, 'mai.{}.jpg'.format(i))
        shutil.copyfile(src, dst)
    else: 
        src = fnames[i]
        dst = os.path.join(test_mai_dir, 'mai.{}.jpg'.format(i))
        shutil.copyfile(src, dst)

fnames = glob.glob("./Aligned/橋本奈々未/*")
train_len = math.floor(len(fnames) * 0.7)
for i in range(len(fnames)):
    if i < train_len:
        src = fnames[i]
        dst = os.path.join(train_nanase_dir, 'nanami.{}.jpg'.format(i))
        shutil.copyfile(src, dst)
    else: 
        src = fnames[i]
        dst = os.path.join(test_nanase_dir, 'nanami.{}.jpg'.format(i))
        shutil.copyfile(src, dst)

fnames = glob.glob("./Aligned/西野七瀬/*")
train_len = math.floor(len(fnames) * 0.7)
for i in range(len(fnames)):
    if i < train_len:
        src = fnames[i]
        dst = os.path.join(train_nanami_dir, 'nanase.{}.jpg'.format(i))
        shutil.copyfile(src, dst)
    else: 
        src = fnames[i]
        dst = os.path.join(test_nanami_dir, 'nanase.{}.jpg'.format(i))
        shutil.copyfile(src, dst)

# Augmenting the Data
names = ["asuka","mai","erika","nanami","nanase"]
for name in names:
    in_dir = "./Input_Data_Rotate/train/"+name+"/*"
    out_dir = "./Input_Data_Rotate/train/"+name
    in_jpg=glob.glob(in_dir)
    img_file_name_list=os.listdir("./Input_Data_Rotate/train/"+name+"/")
    for i in range(len(in_jpg)):
        img = cv2.imread(str(in_jpg[i]))
        # Rotate Images
        for ang in [-10,0,10]:
            img_rot = ndimage.rotate(img,ang)
            img_rot = cv2.resize(img_rot,(150,150))
            fileName=os.path.join(out_dir,str(i)+"_"+str(ang)+".jpg")
            cv2.imwrite(str(fileName),img_rot)
            # Threshold
            img_thr = cv2.threshold(img_rot, 100, 255, cv2.THRESH_TOZERO)[1]
            fileName=os.path.join(out_dir,str(i)+"_"+str(ang)+"thr.jpg")
            cv2.imwrite(str(fileName),img_thr)
            # Filter Images
            img_filter = cv2.GaussianBlur(img_rot, (5, 5), 0)
            fileName=os.path.join(out_dir,str(i)+"_"+str(ang)+"filter.jpg")
            cv2.imwrite(str(fileName),img_filter)
        