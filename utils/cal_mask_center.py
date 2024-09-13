import cv2


def cal_center(mask):
    # 计算前景区域的几何中心
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] == 0:
        return None
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    center = (center_x, center_y)

    return center


def inside_rect(center, rectangle):
    # 判断几何中心是否在给定的矩形区域内
    center_x, center_y = center
    rect_x1, rect_y1, rect_x2, rect_y2 = rectangle
    if rect_x1 <= center_x <= rect_x2 and rect_y1 <= center_y <= rect_y2:
        inside_rectangle = True
    else:
        inside_rectangle = False

    return inside_rectangle


def visualize_center_rect(mask, center, rectangle):
    # 可视化几何中心和矩形框
    visual_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.circle(visual_image, center, 5, (0, 255, 0), -1)
    rect_x1, rect_y1, rect_x2, rect_y2 = rectangle
    cv2.rectangle(visual_image, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), 2)
    return visual_image


if __name__ == '__main__':
    mask_image = cv2.imread("images/179_road.png", cv2.IMREAD_GRAYSCALE)  # 请将 "mask_image.png" 替换为您的二值掩模图像路径
    rectangle_coords = (740, 780, 1645, 995)  # 请替换为您的矩形坐标(x1,y1,x2,y2)

    center = cal_center(mask_image)
    in_rect = inside_rect(center, rectangle_coords)
    visual_image = visualize_center_rect(mask_image, center, rectangle_coords)

    if in_rect:
        print("几何中心坐标:", center, "在矩形区域内")
    else:
        print("几何中心坐标:", center, "不在矩形区域内")

    # 显示可视化图像
    cv2.imshow("Visual Image", visual_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
