# import splitfolders
import random
import os
import shutil
from icecream import ic
random.seed(1000)

def mkpth(destination):
    if not os.path.exists(destination):
        print(destination)
        os.makedirs(destination)
    return destination
topdir = './images/generated_audio/ljspeech_waveglow'
    
dirs =[os.path.join(topdir,i) for i in os.listdir(topdir)]
ic(dirs[0])
tot =len(dirs)
ic(len(dirs))
ratio = (0.7,0.2,0.1)
num_trn = round(ratio[0] * tot)
num_tst = round(ratio[1] * tot)
num_val = tot - (num_trn+num_tst)
ic(num_trn,num_tst,num_val,tot)
random.shuffle(dirs)
ic(dirs[0])
split = {'train':num_trn,'test':num_tst,'validation':num_val}


k = 0
for i,j in split.items():
    destination = os.path.join(topdir,i)
    mkpth(destination)
    ic(i,j)
    for f in dirs[k:j+k]:
        dest = shutil.move(f, destination)
    k += j