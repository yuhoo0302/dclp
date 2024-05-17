# %%
import cv2
import json
import onnxruntime
import numpy as np
import torch
from utils import batched_nms_rotated,nms_rotated,trans_result_with_angle_mod,get_retangle_boxes_mod
# %%
# 鼻唇冠状切面：normal lip view（NLV）
# 上牙槽切面： normal alveolar and palate view (NAPV)
# 异常鼻唇切面：cleft lip view (CLV)
# 异常上牙槽切面：cleft alveolar and palate view(CAPV)
# 上唇： Upper lip （UL）
# 鼻子：nose (N)
# 鼻孔：nostrils (No)
# 下唇：lower lip (LL)
# 下巴：chin  
# 异常上唇：cleft lip (CL)
# 异常上牙槽：cleft alveolar  (CA)
# 异常继发腭：cleft palate (CP)

# %%

def get_model(model_path, is_cuda=True):
    """
    加载模型
    """
    if is_cuda:
        model = onnxruntime.InferenceSession(
            model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print("##################load model in gpu")
    else:
        model = onnxruntime.InferenceSession(model_path,providers=['CPUExecutionProvider'])
        print("###################load model in cpu")
    return model


# %% 切面分类模型
class SpClsInferModule:
    def __init__(self, ) -> None:
        model_path="/home/weki/clp/models/cls_cel4.onnx"
        self.models = get_model(model_path)

    def center_crop(self, img, output_size):
        """
        对图像进行中心裁剪

        参数:
        - img: NumPy 数组，表示图像
        - output_size: 期望的输出尺寸 (h, w)

        返回:
        - 裁剪后的图像: NumPy 数组
        """

        # 获取图像原始尺寸
        img_height, img_width = img.shape[:2]

        # 计算裁剪的左上角坐标
        left = (img_width - output_size[1]) // 2
        top = (img_height - output_size[0]) // 2

        # 计算裁剪的右下角坐标
        right = (img_width + output_size[1]) // 2
        bottom = (img_height + output_size[0]) // 2

        # 进行中心裁剪
        cropped_img = img[top:bottom, left:right, :]

        return cropped_img

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)    
        img = self.center_crop(img, (224,224))
        img = (img/255.)
        mean_ary = np.array([0.485, 0.456, 0.406])#.reshape(1,1,3)
        std_ary = np.array([0.229, 0.224, 0.225])#.reshape(1,1,3)
        img = (img -  mean_ary)/std_ary
        img = np.transpose(img,(2,0,1))
        img = img[np.newaxis,::].astype(np.float32)
        return img
    
    def process(self, image_path) -> dict:
        img=cv2.imread(image_path)
        img = self.preprocess(img)
        outputs = self.models.run(None,{"input": img}) 
        outputs = torch.as_tensor(np.array(outputs), dtype=torch.float32).reshape(-1) # n
        outputs =  torch.softmax(outputs, dim=0)
        sp_scores, sp_classes = outputs.topk(1)
        sp_scores, sp_classes = sp_scores.tolist(), sp_classes.tolist()
        sp_class = cls_map[sp_classes[0]] if sp_classes[0] in cls_map else "unk"
        sp_score=sp_scores[0]
        return {'sp_score': sp_score, 'sp_classe': sp_class}
    
    def process_batch(self,image_paths):
        results=[]
        for image in image_paths:
            result=self.process(image)
            results.append((result["sp_classe"],result["sp_score"]))
        return results


# %% 切面分类映射表
cls_map={0:"NAPV",1:"NLV",2:"CLV",3:"CAPV"}
# # %% 分类模型测试
# cls_model=SpClsInferModule()
# print(cls_model.process("/data/lr/prenatal_project_py/gen/clp_val/image_level_inner_test/val_normal/上牙槽突切面/FT-1st_M3_20200627092817_13_1_60.png"))

# %%
class DetInferModule:
    def __init__(self, ) -> None:
        model_path="/home/weki/clp/models/det_211.onnx"
        self.model = get_model(model_path)
        self.conf_thres = 0.3  
        self.nms_thres = 0.3
        self.batch_size = 1
        self.input_size = (480, 480)
        self.grids_const, self.strides_const = self.grid_generator()

    def grid_generator(self):
        grids = []
        strides = []
        strides_ori = [8, 16, 32]
        # hw = [[52, 52], [26, 26], [13, 13]]  # for 416x416
        hw = [[self.input_size[1] // stride, self.input_size[0] // stride]
              for stride in strides_ori]
        # print(hw)
        for (hsize, wsize), stride in zip(hw, strides_ori):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        grids = torch.cat(grids, dim=1).type(torch.float32)
        strides = torch.cat(strides, dim=1).type(torch.float32)
        return grids, strides

    def preprocess(self, images, device="cpu"):
        imgs = np.stack(images)
        imgs = torch.as_tensor(imgs, dtype=torch.float32, device=device)
        imgs = imgs.permute(0, 3, 1, 2)

        padded_img = torch.ones((self.batch_size, 3, self.input_size[0],
                                 self.input_size[1])) * 114.0
        det_ratios = min(1.0 * self.input_size[0] / imgs.shape[2],
                         1.0 * self.input_size[0] / imgs.shape[3])

        det_img = torch.nn.functional.interpolate(
            imgs,
            size=(int(imgs.shape[2] * det_ratios),
                  int(imgs.shape[3] * det_ratios)),
            mode='bilinear',
            align_corners=True)
        padded_img[:, :, :int(det_img.shape[2]), :int(det_img.shape[3]
                                                      )] = det_img
        return padded_img, det_ratios,

    def to_numpy(self, tensor):
        return tensor.detach().numpy(
        ) if tensor.requires_grad else tensor.detach().cpu().numpy()

    def decode_outputs(self, outputs):
        outputs[..., :2] = (outputs[..., :2] +
                            self.grids_const) * self.strides_const
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * self.strides_const
        return outputs

    def det_postprocess(self,
                        prediction,
                        num_classes,
                        conf_thre=0.1,
                        nms_thre=0.3,
                        class_agnostic=False):
        """
        Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred, glide_3)
        """
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:,
                                                          8:8 + num_classes],
                                               1,
                                               keepdim=True)

            # Detections ordered as (x1, y1, x2, y2, obj_conf,class_conf,class_pred, glid_3)
            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >=
                         conf_thre).squeeze()
            detections = torch.cat((image_pred[:, :5], class_conf,
                                    class_pred.float(), image_pred[:, 5:8]), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            if class_agnostic:
                nms_out_index = nms_rotated(
                    detections,
                    iou_threshold=nms_thre,
                )
            else:
                nms_out_index = batched_nms_rotated(
                    detections,
                    iou_threshold=nms_thre,
                )

            detections = detections[nms_out_index]
            # convert to rotate box
            # detections[:, 0] += detections[:, 0]*detections[:, 5]
            # detections[:, 1] -= detections[:, 1]*detections[:, 6]

            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output

    def convert_out(self, output, ratio, img_h, img_w, conf_thre=0.1):
        if output is None:
            return output
        cls_conf = torch.mean(output[:, 4:6], dim=1,
                              keepdim=True)  # obj_conv, class_conf
        cls_pred_glide_3 = output[:, 6:]  # cls, glide3
        cls_pred_glide_3[:, 1:3][cls_pred_glide_3[:, 1:3] < 0.05] = 0
        bboxes = output[:, :4]
        bboxes /= ratio


        output = torch.hstack((bboxes, cls_conf, cls_pred_glide_3))
        conf_mask = (output[:, 4]  >= conf_thre*1.5).squeeze()
        output = output[conf_mask].reshape(-1,9)
        # box4, cls_conf, cls_pred, glide3 == 9

        # output[:,5] += 1
        output = trans_result_with_angle_mod(output)
        return output

    def preprocess_v2(self, images, device="cpu"):
        images = images[0].astype("float32")
        r = min(480 / images.shape[0], 480 / images.shape[1])
        unpad_w = round(r * images.shape[1])
        unpad_h = round(r * images.shape[0])
        re = cv2.resize(images, (unpad_w, unpad_h),
                        interpolation=cv2.INTER_LINEAR)
        padded_img = np.full((480, 480, 3), 114, dtype="float32")
        padded_img[:int(re.shape[0]), :int(re.shape[1]), :] = re
        padded_img = torch.as_tensor(padded_img, dtype=torch.float32).view(
            1, 480, 480, 3).permute(0, 3, 1, 2)
        return padded_img, r

    def process(self, image_path) -> dict:
        img = cv2.imread(image_path)
        det_img, ratio = self.preprocess_v2([img], device="cpu")
        inputs = {
            self.model.get_inputs()[0].name: self.to_numpy(det_img)
        }
        outputs = self.model.run(None, inputs)[0]
        outputs = torch.from_numpy(outputs)
        outputs = self.decode_outputs(outputs)
        det_output = self.det_postprocess(outputs, num_classes= 211, \
                                conf_thre=self.conf_thres, nms_thre=self.nms_thres, class_agnostic=False)
        img_h, img_w = img.shape[:2]
        outputs = self.convert_out(det_output[0],
                                   ratio,
                                   img_h,
                                   img_w,
                                   conf_thre=self.conf_thres)
        outputs=self.restrict_output(outputs)
        self.visualize(img, outputs)
        return outputs

    def restrict_output(self, outputs):
        cls_list=outputs[:,4]

        unique=np.unique(cls_list)
        filter_list=[]
        for i in unique:
            current_result=outputs[outputs[:,4]==i]
            sorted_result=current_result[current_result[:,5].argsort()]
            if i in double_list:
                sorted_result=sorted_result[:2]
            else:
                sorted_result=sorted_result[:1]

            filter_list.append(sorted_result)

        filter_out=np.concatenate(filter_list)
        return filter_out
    def process_batch(self,image_paths):
        result_list=[]
        for image in image_paths:
            result_list.append(self.process(image))
        return result_list
            
    


    def visualize(self,image, outputs, save_path="output.png"):
        for output in outputs:
            x1,y1,x2,y2,cls,score,rotate=output


            center=((x1+x2)/2,(y1+y2)/2)   
            width=x2-x1
            height=y2-y1
            rect=(center,(width,height),-1*rotate)

            box=cv2.boxPoints(rect)
            box=np.intp(box)
            cls=int(cls)
            cv2.polylines(image, [box], isClosed=True, color=(0, 200, 0), thickness=2)
            if cls in det_map["normal"]:
                cv2.putText(image,det_map["normal"][cls],(int(x1),int(y1)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            elif cls in det_map["abnormal"]:
                cv2.putText(image,det_map["abnormal"][cls],(int(x1),int(y1)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
            
            else:
                print(f"cls id :{cls} not in map")
        
        if save_path:
            cv2.imwrite(save_path,image)
        
        return image



# %%
# det_model= DetInferModule()
det_map={"normal":{62:"Upper lip(UL)",63:"nose(N)",64:"Lower lip(LL)",65:"chin",61:"nostrils(No)"}, "abnormal":{164:"cleft lip(CL)",163:"cleft alveolar(CA)",166:"cleft palate(CP)",167:"CL_ROI",162:"CLP_ROI"}}
double_list=[61]

# print(det_model.process('/data/lr/prenatal_project_py/gen/clp_val/patient_level_external/abnormal/cases_30/szslyy-yc-24w-wyq-2-1777_0.png'))



