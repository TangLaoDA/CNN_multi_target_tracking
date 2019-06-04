import os
import numpy as np
import matplotlib.image as image
import PIL.Image as pimage
import PIL.ImageDraw as imagedraw
import random as ran
import cv2

dir = "background"
#
# images = []
# coords = []
# confidences = []
x = 1

for filename in os.listdir(dir):
    print(filename)
    # 从系统直接读进来的filename包含了整体文件名（??????.jpg or ?????.png,....）
    background = pimage.open("{0}/{1}".format(dir, filename))  # 批量读出要处理的图片
    shape = np.shape(background)
    # print(len(shape))
    if len(shape) == 3:
        background = background
    else:
        continue
    # print(shape)
    background_resize = background.resize((300, 300))

    name = np.random.randint(1, 21)
    # 直接打开的文件 文价名字和格式是分开的
    cup = pimage.open("yellow/{0}.png".format(name))  # 批量读出要处理的图片

    # rot_cup = cup.rotate(np.random.randint(-90, 90))

    ran_w = np.random.randint(60, 200)
    ran_h = ran_w
    img_new = cup.resize((ran_w, ran_h))  # 将要处理的图片按背景图比例缩放
    # print("ran_w and ran_h:",ran_w,ran_h)

    ran_x1 = np.random.randint(0, 300 - ran_w)
    ran_y1 = np.random.randint(0, 300 - ran_h)
    # print("ran_x and ran_y:",ran_x1,ran_y1,"\n")

    r, g, b, a = img_new.split()
    background_resize.paste(img_new, (ran_x1, ran_y1), mask=a)  # 将缩放后的图片按起始位置贴到背景图上
    # background_resize.show()
    ran_x2 = ran_x1 + ran_w
    ran_y2 = ran_y1 + ran_h
    print(ran_x1,ran_x2,ran_y1,ran_y2)
    print(type(background_resize))
    img = cv2.rectangle(np.array(background_resize),(ran_x1,ran_y1),(ran_x2,ran_y2),color=0xff00)
    cv2.imshow("yello",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    background_resize.save("JsamplePic/{0}{1}.png".
                           format(x, "." + str(0) + "." + str(0) + "." + str(0) + "." + str(
        0) + "." + "0"))  # 保持到目标位置
    x = x + 1











