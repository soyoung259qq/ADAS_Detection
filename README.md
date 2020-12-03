# ADAS_Detection
자율주행을 위한 영상 기반 객체 검출
# Train, Test data
* Train : coco 2017 dataset, 80종류 객체에 대한 classification, bounding box, segmentation labeling 된 데이터
![coco](https://user-images.githubusercontent.com/34363323/100237007-b2a68700-2f71-11eb-883f-d380d279720c.png)
* Test : kitti dataset, 자동차 주행 상황 데이터
![kitti](https://user-images.githubusercontent.com/34363323/100237131-d8cc2700-2f71-11eb-9c46-39e69300869e.png)
# Model
RetinaNet 모델 적용 </br>
<img width="845" alt="Screen_Shot_2020-06-07_at_4 22 37_PM" src="https://user-images.githubusercontent.com/34363323/100236987-b0442d00-2f71-11eb-9093-65a0666ed993.png">
# Result
![그림1](https://user-images.githubusercontent.com/34363323/100236996-b1755a00-2f71-11eb-9e12-b70ef6ac4917.png)
![그림2](https://user-images.githubusercontent.com/34363323/100236999-b1755a00-2f71-11eb-92f8-19340e11d5ca.png)
![그림3](https://user-images.githubusercontent.com/34363323/100237002-b20df080-2f71-11eb-8b0f-bc1bb02311c0.png)
![그림4](https://user-images.githubusercontent.com/34363323/100237004-b2a68700-2f71-11eb-971c-ce3090a0589d.png)
# Reference
<a href="https://cocodataset.org/#home"> COCO Dataset </a> </br>
<a href="http://www.cvlibs.net/datasets/kitti/eval_tracking.php"> Kitti tracking dataset </a></br>
<a href="https://keras.io/examples/vision/retinanet"> Reference code </a> </br>
