import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    # ML-DETR 最终模型配置（论文 Table II, Order 11；13.9 GFLOPs / 9.06M params, nc=1）
    model = RTDETR('ultralytics/cfg/models/yolo-detr/yolov8-detr-goldyolo-asf-RGCSPELAN.yaml')
    # model.load('') # loading pretrain weights预训练权重指定
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=8,
                workers=4,
                device='0',
                # resume='', # last.pt path
                project='runs/train',
                name='ZY',
                )