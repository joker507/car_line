'''左右车道线检测'''
import cv2
import numpy as np
import math

def Show(image):
    # 显示图片
    cv2.imshow("tast", image)
    cv2.waitKeyEx(0)
    cv2.destroyAllWindows()


# 车道颜色选选择
def select_rgb_lines(image,colors):
    '''
    color:[[lower,upper],[lower,upper],...]

    sample: [
        [[200, 200, 200],[255, 255, 255]],
        ...
        ]
    '''
    all_mask = np.zeros(image.shape[:2],dtype=np.uint8)#创建一个黑色画布，用作在上面添加使用颜色检测的车道线。
    for color in colors: #将图片中每一个符合设置在颜色阈值内的区域保留，加到黑色画布上面
        lower = np.uint8(color[0])
        upper = np.uint8(color[1])
        mask = cv2.inRange(image, lower, upper)  # 在范围的就为255 else 0
        all_mask = cv2.bitwise_or(all_mask,mask)

    return cv2.bitwise_and(image, image, mask=all_mask) # 原图像的每个像素和all_mask的对应像素点相与，相当于在原图上扣出来符合阈值的像素



# 灰度图
def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 灰度图公式Grey = 0.299*R + 0.587*G + 0.114*B


# 平滑：高斯滤波
def apply_smoothing(image, kernel_size=15):
    '''
    创建一个符合高斯分布的核，大小为kernel_size
    使用该核对每一个像素点进行计算，达到平滑消除高斯噪声的目的
    '''
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# 边缘检测
def detect_edges(image, low_threshold=50, high_threshold=150):
    '''
    使用Sobel算子，进行边缘检测。
    原理：1. 高斯去噪
         2. Sobel梯度计算
         3. 非极大值抑制
         4. 滞后阈值
    '''
    return cv2.Canny(image, low_threshold, high_threshold)


# 利益区域选择
def filter_region(image, vertices):
    """
    选择填充区域
    根据vertices点，原图在vertices范围内的内容
    """
    mask = np.zeros_like(image)  # 构建一幅尺寸一样的图以0填充
    if len(mask.shape) == 2:  # 图片为单通道的时候也就是灰度图
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])

    return cv2.bitwise_and(image, mask)


def select_region(image):
    """
   选择RIO区域
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)


# 霍夫直线检测
def hough_lines(image):
    '''rho 为精度
        thera 也是精度
        相当于建立了一个表格行为rho列为theta,根据精度确定大小
        thresh 为阈值只有累加器大于此值的时候才认为是一条直线
        minLineLength 为长度阈值只有大于20长度的认为是一条直线
        masLineGap 为间隔阈值，两条直线的间隔小与300认为是一直线'''
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)


# 左右单车道线各自拟合
def average_slope_intercept(lines):
    """通过计算每天加权平均值拟合一条直线"""
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)
    if not lines.all():  # 含有false
        return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:  # 如果是竖直。忽略
                continue  # ignore a vertical line
            slope = (y2 - y1) / (x2 - x1)  # 斜率
            intercept = y1 - slope * x1  # 纵截距
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)  # 直线长度为权重
            if slope < 0:  # 斜率判断为左车道线
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:  # 右车道线
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # 计算拟合左右车道线
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(
        left_weights) > 0 else None  # 求出加权平均值得K和b
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)


def use_kb_make_line_points(y1, y2, line):
    """
    利用斜率截距计算出直线的两点方便画线
    """
    if line is None:
        return None

    slope, intercept = line
    if slope == 0:  # 当直线斜率为0时候:
        return ((0, y1), (100, y2))
    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    '''
    拟合直线主函数
    1. 得出左右车道线
    2。 计算直线端点
    '''
    left_lane, right_lane = average_slope_intercept(lines)  # '->截距和斜率'

    y1 = image.shape[0]  # bottom of the image
    y2 = y1 * 0.6  # slightly lower than the middle

    left_line = use_kb_make_line_points(y1, y2, left_lane)
    right_line = use_kb_make_line_points(y1, y2, right_lane)  # 得到两个点

    return left_line, right_line


def draw_lane_lines(image, lines, color=[0, 0, 255], thickness=20):
    # 画线
    line_image = np.zeros_like(image)  # 创建画板

    if not lines:
        return None
    for line in lines:
        if line is not None:  # 如果有线就画
            cv2.line(line_image, *line, color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)
q

def draw_tow_midline(image, lines, color=[0, 255, 0], thickness=10):
    # 两个车道线的中线
    # ?如果有一遍车道线检测不出来的时候将沿用上一帧的中线？
    '''return image , data'''
    line_image = np.zeros_like(image)

    left_line = lines[0]
    right_line = lines[1]
    if left_line and right_line:
        pt1 = ((left_line[0][0] + right_line[0][0]) // 2, (left_line[0][1] + right_line[0][1]) // 2)
        pt2 = ((left_line[1][0] + right_line[1][0]) // 2, (left_line[1][1] + right_line[1][1]) // 2)
        cv2.line(line_image, pt1, pt2, color, thickness)
    elif right_line:
        cv2.line(line_image, *right_line, color, thickness)

    elif left_line:
        cv2.line(line_image, *left_line, color, thickness)

    if (pt1[1] - pt2[1]) == 0:
        angle = 0
    else:
        angle = math.atan((pt1[0] - pt2[0])/(pt1[1] - pt2[1]))
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0), angle

class Car_lines():
    '''
    img:一幅通道为RGB的图像
    '''
    def __init__(self):
        pass
    def search(self,img,colors=[[[200, 200, 200],[255, 255, 255]],[[0, 110, 110],[100, 255, 255]]]):
        #     # 黄色
        #     lower = np.uint8([0, 110, 110])
        #     upper = np.uint8([100, 255, 255])
        #     # 白色RGB
        #     lower = np.uint8([200, 200, 200])
        #     upper = np.uint8([255, 255, 255])

        white_yellow = select_rgb_lines(img,colors)  # 颜色选择
        gray = convert_gray_scale(white_yellow)  # 转换灰度图
        smooth_gray = apply_smoothing(gray)  # 高斯平滑
        edges = detect_edges(smooth_gray)  # 边缘检测
        roi = select_region(edges)  # 选择车道线位置，要知道大概位置
        lines = hough_lines(roi)  # 直线检测
        l_r_lines = lane_lines(img, lines)  # 拟合直线
        result = draw_lane_lines(img, l_r_lines)  # 画线
        result, data = draw_tow_midline(result, l_r_lines)  # 画中线
        return result, data

if __name__ == '__main__':
    cap = cv2.VideoCapture("test_videos/challenge.mp4")  # 1280*720 25f/s #开启摄像头
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('out_video/output.mp4', fourcc, 20.0, (width, height))  # 注意设置的大小和原来视频的要相同

    search = Car_lines()#实例化
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            result,data = search.search(frame,colors=[[[200, 200, 200],[255, 255, 255]],[[0, 110, 110],[100, 255, 255]]]) #调用寻找图片中的车道线
            print(data)
            out.write(result)  # 保存视频
            cv2.imshow("frame", result)  # 显示图片
            if cv2.waitKeyEx(40) & 0xFF == ord('1'):  # 退出键 #25fps/s
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
