import base64
import os
from io import BytesIO

import obtools as ob
import cv2

from PIL import Image
from model_infer import SpClsInferModule, DetInferModule


class DCLP:
    def __init__(self) -> None:
        self.cls_model = SpClsInferModule()
        self.det_model = DetInferModule()

    def diaglosis(self, cls_results):
        diagnosis = None
        cls_list = [item[0] for item in cls_results]
        cls_set = set(cls_list)

        exists_BC = any(element for element in cls_set if element in set(["NLV", "CLV"]))

        exists_SYC = any(element for element in cls_set if element in set(["NAPV", "CAPV"]))

        if exists_BC and exists_SYC:

            if "CLV" in cls_list and "CAPV" in cls_list:
                diagnosis = {"Abnormal": "CLP"}

            if "CLV" not in cls_list and "CAPV" in cls_list:
                diagnosis = {"Normal": "Normal"}

            if "CLV" in cls_list and "CAPV" not in cls_list:
                diagnosis = {"Abnormal": "CL"}

            if "CLV" not in cls_list and "CAPV" not in cls_list:
                diagnosis = {"Normal": "Normal"}

        if exists_BC and not exists_SYC:
            diagnosis = {"Uncertain": "without alveolar&palate view"}

        if not exists_BC and exists_SYC:
            diagnosis = {"Uncertain": "without lip view"}

        if not exists_BC and not exists_SYC:
            diagnosis = {"invalid": "no lip view and no alevolar&palate view"}
        return diagnosis

    def convert_normal_planes(self,cls_results):
        
        return [i[0]in[0,1]for i in cls_results] # NLV,NAPV
    
    def detect_and_classify(self, images_path):

        p_result = {"Diagnosis": None, "Support": [], "Det_imgs": []}

        cls_results = self.cls_model.process_batch(images_path)
        det_results = self.det_model.process_batch(images_path)
        normal_list=self.convert_normal_planes(cls_results)
        det_results=self.det_model.visualize_batch(images_path,det_results,normal_list)


        p_result["Support"] = cls_results
        p_result["images"] = det_results
        p_result["Diagnosis"] = self.diaglosis(cls_results)

        return p_result

    def visualize(self, result,p_images):
        plot_img_list = []
        for index,(plot_img, pred) in enumerate(zip(result["images"], result["Support"])):
            plot_img = ob.cv2Img_AddChinese(plot_img, "Plane: " + pred[0] + " conf: " + str(round(pred[1], 3)),
                                            (40, 40))

            infor = [k + ": " + result["Diagnosis"][k] for k in result["Diagnosis"].keys()][0]
            plot_img = ob.cv2Img_AddChinese(plot_img, infor, (40, 80))
            cv2.imwrite(os.path.basename(p_images[index]),plot_img)

            plot_img_list.append(self.convert2base64(plot_img))

        result["images"] = plot_img_list

    def convert2base64(self, image):
        #  把image转换成base64
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image = base64.b64encode(buffered.getvalue()).decode()

        return image


file_ext = ["jpg", "png", "jpeg", "bmp"]
dclp = DCLP()


def detect_and_classify(images_path="/home/weki/clp/test_sample/CL3"):
    print('begin')
    p_images = os.listdir(images_path)
    p_images = [os.path.join(images_path, item) for item in p_images if item.split(".")[-1] in file_ext]

    result = dclp.detect_and_classify(p_images)
    dclp.visualize(result,p_images)

    return result


if __name__ == "__main__":
    result = detect_and_classify()
    print(result["Diagnosis"])
    print(result["Support"])
