自动驾驶视角道路锥形桶安全锥检测数据集 柱形图车辆视角路面锥形桶等交通导流实例分割数据集 yolo分割模型 道路导流实例分割系统界面![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f1130b42f8da46a4aef7404c23fca7a0.png)



车辆视角路面锥形桶等交通导流实例分割数据集，3726张，yolo，voc，coco三种标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fccf915e6df644b784bdd1fdff413b54.png)

数据集统计信息![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d730872a63f446f1bb41944164e24de3.png)

====================
图像尺寸:800*800
类别数量: 3 类
训练集图像:3300; 验证集图像:363; 测试集图像:63
总图像数: 3726

类别分布:![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/47b160b0a66345ae9ce09646965f6052.png)

类别名称 | 包含该类别的图像数 | 总标注数
----------------------------------------![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b94a98463f624cbdb5cd71a98cfbc3c7.png)

barrel - 圆形路桶，2187,10430
channelizer - 交通导流，786,3442
cone - 锥形桶，620,2801
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c4383ec693fc427280901107678bcb5a.png)


模型与源码：yolov11n-seg训练，80个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/be42faa765c343c5b03c8bdb216ac5cc.png)



**qt界面系统**：提供全部源代码，运行界面采用pyqt5编写，可支持图像，视频，摄像头的实例分割，本项目已经训练好模型，参照教程配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e27a3d38fdab4cbcb23deab79ffefc72.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/81a91e0fdf584baf9a6de44fbf4e0172.png)
# 车辆视角锥形桶等路面导流设施实例分割系统 YOLOv11-seg 完整工程
## 1. 数据集配置文件 `road_barrier.yaml`
```yaml
path: ./road_barrier_dataset
train: train/images
val: val/images
test: test/images

names:
  0: barrel         # 防撞桶
  1: channelizer    # 导流标/隔离桩
  2: cone           # 交通锥
```

## 2. 模型训练代码 `train.py`
```python
from ultralytics import YOLO

if __name__ == "__main__":
    # 加载YOLOv11分割预训练权重
    model = YOLO("yolo11n-seg.yaml").load("yolo11n-seg.pt")
    # 训练参数
    results = model.train(
        data="road_barrier.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        save=True,
        pretrained=True,
        patience=10
    )
    print("训练结束，权重保存在 runs/segment/train/weights/best.pt")
```

## 3. PyQt5可视化GUI系统 `main.py`（）
```python
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog,
                             QLabel, QTableWidget, QTableWidgetItem, QCheckBox, QRadioButton)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO

class DetThread(QThread):
    signal_img = pyqtSignal(QPixmap)
    signal_table = pyqtSignal(list)
    def __init__(self, model, img_path, seg_mode):
        super().__init__()
        self.model = model
        self.img_path = img_path
        self.seg_mode = seg_mode
    def run(self):
        res = self.model.predict(self.img_path, conf=0.5, task="segment" if self.seg_mode else "detect")
        img = cv2.imread(self.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes_info = []
        for r in res:
            for box in r.boxes:
                cls_name = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                boxes_info.append([cls_name, f"{conf:.2f}", int(xyxy[0]), int(xyxy[1])])
        img_result = res[0].plot()
        h,w,c = img_result.shape
        qimg = QImage(img_result.data, w, h, c*w, QImage.Format_RGB888)
        self.signal_img.emit(QPixmap.fromImage(qimg))
        self.signal_table.emit(boxes_info)

class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLO11的车辆视角锥形桶等路面导流实例分割系统")
        self.resize(1600,900)
        self.model = YOLO("runs/segment/train/weights/best.pt")
        self.seg_mode = True
        self.init_ui()
    def init_ui(self):
        # 功能按钮
        self.btn_img = QPushButton("选择图片",self)
        self.btn_img.move(20,120)
        self.btn_img.clicked.connect(self.load_image)
        self.btn_video = QPushButton("选择视频",self)
        self.btn_video.move(130,120)
        self.btn_cam = QPushButton("摄像头",self)
        self.btn_cam.move(240,120)
        self.btn_save = QPushButton("保存结果",self)
        self.btn_save.move(350,120)
        self.check_preview = QCheckBox("实时预览",self)
        self.check_preview.move(460,120)
        self.radio_det = QRadioButton("检测模式",self)
        self.radio_det.move(560,120)
        self.radio_seg = QRadioButton("分割模式",self)
        self.radio_seg.move(660,120)
        self.radio_seg.setChecked(True)
        self.radio_seg.toggled.connect(self.change_mode)
        # 图像显示区域
        self.label_ori = QLabel("原始图像",self)
        self.label_ori.setGeometry(30,200,750,650)
        self.label_res = QLabel("分割结果",self)
        self.label_res.setGeometry(800,200,750,650)
        # 检测结果表格
        self.table = QTableWidget()
        self.table.setGeometry(1580,200,320,400)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["类别","置信度","xmin","ymin"])

    def change_mode(self):
        self.seg_mode = self.radio_seg.isChecked()

    def load_image(self):
        path,_ = QFileDialog.getOpenFileName(self,"选择图片","","Image Files(*.jpg *.png *.jpeg)")
        if not path:return
        ori_pix = QPixmap(path).scaled(self.label_ori.size(),Qt.KeepAspectRatio)
        self.label_ori.setPixmap(ori_pix)
        self.th = DetThread(self.model, path, self.seg_mode)
        self.th.signal_img.connect(self.show_result_img)
        self.th.signal_table.connect(self.fill_table)
        self.th.start()

    def show_result_img(self,pix):
        pix = pix.scaled(self.label_res.size(),Qt.KeepAspectRatio)
        self.label_res.setPixmap(pix)
    def fill_table(self,data):
        self.table.setRowCount(len(data))
        for i,row in enumerate(data):
            for j,val in enumerate(row):
                self.table.setItem(i,j,QTableWidgetItem(str(val)))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWin()
    win.show()
    sys.exit(app.exec_())
```

## 4. 环境安装
```bash
pip install ultralytics opencv-python pyqt5 torch torchvision
```

## 5. 系统功能
1. 图片、视频、车载摄像头实时检测/实例分割
2. 一键切换【目标检测模式 / 实例分割模式】
3. 右侧表格输出类别、置信度、目标坐标
4. 可视化绘制掩码+检测框+类别标签，和截图界面完全一致
5. 支持保存推理结果图片

