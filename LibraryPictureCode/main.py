import requests
import ddddocr
import os
import cv2 as cv

# picture num
num = 100000




def update_ocr_picture():
    ocr = ddddocr.DdddOcr()
    file_list = [os.path.join("./ocr", file_name) for file_name in os.listdir("./picture")]
    print(file_list)
    for file in file_list:
        with open(file, "rb") as f:
            content = f.read()
            result = ocr.classification(content).replace("o", "0").replace("O", "0")
            print(result)
            with open(f"./ocr/{result}.png", "wb") as f:
                f.write(content)

def ocr_verification_code():
    ocr = ddddocr.DdddOcr()
    file_list = [os.path.join(r"./picture", file_name) for file_name in os.listdir("./picture")]
    print(file_list)
    for file in file_list:
        with open(file, "rb") as f:
            content = f.read()
            result = ocr.classification(content).replace("o", "0").replace("O", "0").replace(">","7")
            print(result)
            with open(rf"./ocr/{result}.png", "wb") as f:
                f.write(content)

def get_picture():
    """
    爬取图书馆图片
    :return:
    """
    URL = "http://222-27-188-3-oqhjdanw-ex0-www-webvpn.webvpn.webvpn2.hrbcu.edu.cn/api.php/check"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/97.0.4692.99 Safari/537.36 Edg/97.0.1072.76",
        "Referer": "http://222-27-188-3-oqhjdanw-ex0-www-webvpn.webvpn.webvpn2.hrbcu.edu.cn/web/seat3?area=22&segment"
                   "=1293786&day=2022-1-30&startTime=19:10&endTime=22:00 "
    }

    for i in range(num):
        resp = requests.get(url=URL, headers=headers)
        with open(f"./picture/{i}.png", "wb") as f:
            f.write(resp.content)


def checkout_digital():
    file_list = [os.path.join("ocr", file) for file in  os.listdir("ocr")]
    # print(file_list)
    for file in file_list:
        if file[4:-4].isdigit() and len(file[4: -4]) == 4:
            # print(file[4:-4])
            with open(file, "rb") as fr:
                content = fr.read()
                with open(f"./checkout/{file[4:-4]}.png", "wb") as fw:
                    fw.write(content)
def RGB2GRAY():
    file_name = os.listdir("checkout")
    file_list = [os.path.join("./checkout", file) for file in file_name]
    w_file_list = [os.path.join("./checkout_gray", file) for file in file_name]
    for i in range(len(file_list)):
        img = cv.imread(file_list[i])
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imwrite(w_file_list[i], img_gray)


if __name__ == '__main__':
    # 获取图片
    # get_picture()
    # 识别图片
    # ocr_verification_code()
    # 更新图片
    # checkout_digital()
    RGB2GRAY()