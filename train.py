import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/data.yaml',  # 数据集路径
                cache=False,
                imgsz=640,
                epochs=400,
                batch=4,
                workers=4,
                # device='0,1',
                # resume='',  # 断点续训
                mosaic=1,
                mixup=0.2,
                patience=30, 
                project='runs/train',
                name='exp',
                )