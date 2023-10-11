from controlnet_aux import HEDdetector

class ScribblePreprocessor:
    def __init__(self):
        hed = HEDdetector.from_pretrained("/home/bruce/.cache/huggingface/hub/models--lllyasviel--Annotators/blobs/",filename="5ca93762ffd68a29fee1af9d495bf6aab80ae86f08905fb35472a083a4c7a8fa")
        self.hed = hed
        
    def image_scribble(self, image):
        scribble = self.hed(image)
        return scribble