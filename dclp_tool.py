#%%
from model_infer import SpClsInferModule,DetInferModule

class DCLP:
    def __init__(self) -> None:
        self.cls_model = SpClsInferModule()
        self.det_model = DetInferModule()
    
    def detect_and_classify(self,images):
        
        p_result={"Diagnosis":None,"Support":[],"Det_imgs":[]}

        cls_results=self.cls_model.process_batch(images)
        det_results=self.det_model.process_batch(images)

        p_result["Support"]=cls_results
        p_result["Det_imgs"]=det_results

        cls_list=[item[0] for item in cls_results]
        cls_set=set(cls_list)

        
        exists_BC=any(element for element in cls_set if element in  set(["NLV","CLV"]))

        exists_SYC=any(element for element in cls_set if element in  set(["NAPV","VAPV"]))

        if exists_BC and exists_SYC:
        
            if "CLV" in cls_list and "CAPV" in cls_list:
                p_result["Diagnosis"]={"Abnormal":"CLP"}

            if "CLV" not in cls_list and "CAPV" in cls_list:
                p_result["Diagnosis"]={"Normal":"Normal"}
            
            if "CLV" in cls_list and "CAPV" not in cls_list:
                p_result["Diagnosis"]={"Abnormal":"CL"}
            
            if "CLV" not in cls_list and "CAPV" not in cls_list:
                p_result["Diagnosis"]={"Normal":"Normal"}

        
        if exists_BC and not exists_SYC:
            p_result["Diagnosis"]={"Uncertain":"without alveolar&palate view"}

        if not exists_BC and exists_SYC:
            p_result["Diagnosis"]={"Uncertain": "without lip view"}

        if not exists_BC and not exists_SYC:
            p_result["Diagnosis"]={"invalid":"no lip view and no alevolar&palate view"}


        return p_result

# %%

if __name__=="__main__":
    import glob
    dclp=DCLP()
    p_images=glob.glob("/home/weki/clp/test_sample/alfyyy-yc-bqx-24w/*.png")

    result=dclp.detect_and_classify(p_images)
    print(result["Diagnosis"])