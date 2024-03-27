from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from icecream import ic
from tqdm import tqdm
import numpy as np
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True,nb_workers=56)


def wv(p):  
#    import matplotlib.pyplot as plt
#    from scipy.io import wavfile
#    import os
#    from icecream import ic
#    from tqdm import tqdm
#    import numpy as np
#    import pandas as pd
#    def mkpth(destination):
#        if not os.path.exists(destination):
#            print(destination)
#            os.makedirs(destination)
#        return destination
    # for p in tqdm(dirs):
    f = os.path.basename(p)            
    if p.endswith('wav'):
        topdir = os.path.dirname(p)
        # ic(topdir)
        sv =''
        sv = './images/' + topdir
        mkpth(sv)
        fn = os.path.join(sv,f[:-4])+'.png'
        # if os.path.exists(fn):
        #     pass
        # else:
        samplingfrequency, signaldata = wavfile.read(p)
        pxx, freq, bins, im = plt.specgram(x=signaldata, Fs=samplingfrequency, noverlap=384, NFFT=512)
        # plt.title(f'spec of vowel{p}')
        #plot.xlabel('time')
        #plot.ylabel('freq')
        plt.gcf().set_size_inches(3, 3)
        plt.axis('off')
        # ic(sv)
        plt.savefig(fn,bbox_inches='tight', dpi=100)
        # plt.show()
    return f

if __name__ =='__main__':
    topdir = 'generated_audio'#'./LJSpeech-1.1'
    # destination=dir+'images'
    def mkpth(destination):
        if not os.path.exists(destination):
            print(destination)
            os.makedirs(destination)
        return destination
    ic(topdir)
    dirs =[os.path.join(i[0],j) for i in os.walk(topdir) for j in i[2]] #[i[0] for i in os.walk(topdir)] 
    
    df = pd.DataFrame(dirs,columns=['dirs'])
    df['dest'] = './images/' + df['dirs']
    df['dest'] = df['dest'].apply(lambda x: x[0:-4]+'.png')
    for i,j in enumerate(df['dest']):
        if os.path.exists(j):
            df.drop(i,inplace=True)
    df.reset_index(inplace=True,drop=True)
    _ = df['dirs'].parallel_apply(wv) 