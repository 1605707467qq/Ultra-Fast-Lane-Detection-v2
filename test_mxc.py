from PIL import Image
import numpy as np
import glob,natsort,os
from tqdm import tqdm
for img_path in tqdm(glob.glob('/root/Ultra-Fast-Lane-Detection-v2/Tusimple/clips/*/*/png')):
    print(os.path.dirname(image_path[0])+'/21.png')
# 读取图片并转换成PIL Image对象
    # image_path = natsort.natsorted(glob.glob(img_path+'/*.jpg'))
    # imgs_5 = [Image.open(image_path[i]) for i in [0, 4, 9, 14, 19]]
    # average_image = np.mean([np.array(image) for image in imgs_5], axis=0)
    # average_image = Image.fromarray(average_image.astype(np.uint8))
    # average_image.save(os.path.dirname(image_path[0])+'/21.jpg')
    # # print(os.path.dirname(image_path[0])+'/21.jpg')