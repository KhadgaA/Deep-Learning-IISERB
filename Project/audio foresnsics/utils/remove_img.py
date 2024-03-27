import random
import os
import shutil
from icecream import ic
from tqdm import tqdm
from PIL import Image
random.seed(1000)

def mkpth(destination):
    if not os.path.exists(destination):
        print(destination)
        os.makedirs(destination)
    return destination


    
#
#ic(dirs[0])
#tot =len(dirs)
#ic(len(dirs))
#ratio = (0.7,0.2,0.1)
## num_trn = round(ratio[0] * tot)
## num_tst = round(ratio[1] * tot)
## num_val = tot - (num_trn+num_tst)
## ic(num_trn,num_tst,num_val,tot)
## random.shuffle(dirs)
#ic(dirs[0])

def main(topdir):
    split = {'train':1,'test':1,'validation':1}
    
    for i,j in split.items():
        destination = os.path.join(topdir,i)
    # destination = "/content/New folder/data/jsut_multi_band_melgan/validation"
        dirs =[os.path.join(destination,i) for i in os.listdir(destination)]
        # mkpth(destination)
        ic(i,j)
        for f in tqdm(dirs):
            try:
                png = Image.open(f)
                png.load() # required for png.split()
                try:
                    background = Image.new("RGB", png.size, (255, 255, 255))
                    background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
    
                    background.save(f, 'png', quality=100)
                except:
                    pass
            except:
                os.remove(f)
                ic(f)
            # break
        # dest = shutil.move(f, destination)
    # k += j
if __name__ =='__main__':
    _ = '/data2/dl/DATASETS/audio foresnsics/images/generated_audio/'
    for x  in ['ljspeech_melgan_large','ljspeech_melgan','ljspeech_hifiGAN','ljspeech_full_band_melgan','jsut_multi_band_melgan']:#ljspeech_parallel_wavegan , ljspeech_multi_band_melgan/,'jsut_parallel_wavegan','ljspeech_waveglow'  
        topdir = _ + x
        main(topdir)
    
    