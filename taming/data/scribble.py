from controlnet_aux import HEDdetector

class ScribblePreprocessor:
    def __init__(self):
        hed = HEDdetector.from_pretrained("~/.cache/huggingface/hub/models--lllyasviel--Annotators/snapshots/982e7edaec38759d914a963c48c4726685de7d9/")
        self.hed = hed
        
    def image_scribble(self, image):
        scribble = self.hed(image)
        return scribble