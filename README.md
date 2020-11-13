## 基于霍夫直线检测

## 使用方法：

```python
#实例化：
import car_lines
import cv2
app = car_lines.Car_lines()

#读取图片
img = cv2.imread(img_path)

#设定车道线颜色阈值/默认白、黄色
white_lower = [200, 200, 200]
white_upper = [255, 255, 255]
yellow_lower = [0, 110, 110]
yellow_upper = [100, 255, 255]
colors = [[white_lower,white_upper],[yellow_lower,yellow_upper]]

#调用方法
result, data = app.search(img,colors=colors)
#result:返回绘制好检测到车道线的图片
#data:中心偏转角
```

## 1.0安装环境 
``` pip install -r requirement.txt```

## 数据准备
在test_videos 中放入车道线视频
结果保存在out_video中

## 调用：
``` python car_lines.py```

*** 
## 改进：
1. 增加参数视频方式还是图片方式
2. 嵌入设备中
3. 数据返回调整
