# %%
from controlnet_aux import HEDdetector
import os
import torchvision
from PIL import Image
from tqdm import tqdm

class ScribblePreprocessor:
    def __init__(self):
        hed = HEDdetector.from_pretrained("/home/bruce/.cache/huggingface/hub/models--lllyasviel--Annotators/blobs/",filename="5ca93762ffd68a29fee1af9d495bf6aab80ae86f08905fb35472a083a4c7a8fa")
        self.hed = hed
        
    def image_scribble(self, image):
        scribble = self.hed(image)
        return scribble

    def process(self, list_file_path, output_dir, dataset_root="datasets/Sketchy", start=0):
        #outpur_dir : datasets/Sketchy/scribble
        print(self.hed.netNetwork.to("cuda"))
        log_file =open("logs/scribble_process.txt","+a")
        with open(file=list_file_path) as f:
            lines = f.readlines()
            for i, line in tqdm(enumerate(lines)):
                if i <start:
                    continue
                file_path = line.split(" ")[0]
                log_file.write(file_path+"\n")
                if i%100 ==0:
                    log_file.flush()
                    
                image = Image.open(os.path.join(dataset_root,file_path))
                scribble = self.invert_color(self.image_scribble(image))
                # print(scribble)
                # torchvision.utils.save_image(scribble, os.path.join(output_dir, file_path))
                self.save_image(scribble, output_dir, file_path)
                
    
    def invert_color(self, im):
        im_inverted = im.point(lambda _: 255-_)
        return im_inverted
    
    def save_image(self, image, output_dir, file_path):
        # print(os.path.dirname((os.path.join(output_dir,file_path))))
        if not os.path.isdir(os.path.dirname(os.path.join(output_dir,file_path))):
            os.makedirs(os.path.dirname(os.path.join(output_dir,file_path)),mode=777)
        image.save(os.path.join(output_dir, file_path))

# %%
import sys
import os 
os.chdir("/root/app/ZSE-SBIR")
print(os.getcwd())
root_dir = os.getcwd()

# %%
sp = ScribblePreprocessor()
# sp.process(os.path.join(root_dir,"datasets/Sketchy/zeroshot0/all_photo_filelist_train.txt"), os.path.join(root_dir, "datasets/Sketchy/scribble"))
sp.process("datasets/Sketchy/zeroshot0/all_photo_filelist_train.txt", "datasets/Sketchy/scribble",start=3800)

# %%
