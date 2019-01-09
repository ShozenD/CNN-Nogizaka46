import os, math, shutil, glob

# Path to the directory that will hold the data
base_dir='./Input_Data_NoChange'
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