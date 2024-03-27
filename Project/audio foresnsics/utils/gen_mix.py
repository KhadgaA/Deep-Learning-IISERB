import random
import os
import shutil
from icecream import ic
from tqdm import tqdm
import matplotlib.pyplot as plt
random.seed(1000)

def mkpth(destination):
    if not os.path.exists(destination):
        print(destination)
        os.makedirs(destination)
    return destination
topdir1 = "./images/LJSpeech-1.1/wavs"
topdir2 = "./images/generated_audio/ljspeech_multi_band_melgan"
topdir3='./images/generated_audio/mixed_gen'
    

split = {'train':1,'test':1,'validation':1}

for i,j in split.items():
    destination1 = os.path.join(topdir1,i)
    destination2 = os.path.join(topdir2,i)
    destination3 = os.path.join(topdir3,i)
    mkpth(destination3)
    # destination1, destination2 = topdir1, topdir2
    # destination = "/content/New folder/data/jsut_multi_band_melgan/validation"
    dirs1 =[os.path.join(destination1,i) for i in os.listdir(destination1)]
    dirs2 =[os.path.join(destination2,i) for i in os.listdir(destination2)]
    # ic(dirs2)
    # mkpth(destination)
    # ic(i,j)
    for i,j in tqdm(enumerate(zip(dirs1,dirs2))):
        f1,f2 = j
        #ic(f1,f2)
        imgr = plt.imread(f1)
        imgf = plt.imread(f2)
        imgm = (imgr + imgf )*0.5
        
        plt.gcf().set_size_inches(3, 3)
        plt.axis('off')
        plt.imsave(os.path.join(destination3,f'{i}.png'),imgm, dpi=100)
    # ic(sv)
    # plt.savefig('test',bbox_inches='tight', dpi=100)
    # break
        # break
    # dest = shutil.move(f, destination)
# k += j