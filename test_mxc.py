from PIL import Image
import numpy as np
import glob,natsort,os
from tqdm import tqdm
for img_path in tqdm(glob.glob('/root/Ultra-Fast-Lane-Detection-v2/Tusimple/clips/*/*')):
    # print(os.path.dirname(image_path[0])+'/21.png')
# 读取图片并转换成PIL Image对象
# python test.py configs/tusimple_regy1_6.py --test_model /root/Ultra-Fast-Lane-Detection-v2/logs/20240313_134307_lr_5e-02_b_8/model_best.pth --test_work_dir ./tmp
    image_path = natsort.natsorted(glob.glob(img_path+'/*.jpg'))
    imgs_2 = [Image.open(image_path[i]) for i in [17, 18]]
    image = Image.open(image_path[19])
    average_image = np.mean([np.array(image) for image in imgs_2], axis=0)
    average_image = Image.fromarray(average_image.astype(np.uint8))
    # average_image.save(os.path.dirname(image_path[0])+'/22.jpg')
    merged_image = Image.blend(average_image, image, alpha=0.9)
    # merged_image.save('temp.jpg')
    merged_image.save(os.path.dirname(image_path[0])+'/21.jpg')
    print(os.path.dirname(image_path[0])+'/21.jpg')
    # break
    