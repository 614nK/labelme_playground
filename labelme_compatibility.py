from labelme import PY2
from labelme import QT4
from labelme import __version__
from labelme import utils
import base64
import PIL.Image
import io
import json
import numpy as np
import os.path as osp

def load_image_file(filename):
    try:
        image_pil = PIL.Image.open(filename)
    except IOError:
        logger.error("Failed opening image file: {}".format(filename))
        return

    # apply orientation to image according to exif
    image_pil = utils.apply_exif_orientation(image_pil)

    with io.BytesIO() as f:
        ext = osp.splitext(filename)[1].lower()
        if PY2 and QT4:
            format = "PNG"
        elif ext in [".jpg", ".jpeg"]:
            format = "JPEG"
        else:
            format = "PNG"
        image_pil.save(f, format=format)
        f.seek(0)
        return f.read()
    

def transform_points(points):
    column = points.shape[0]
    transformed_points = np.empty(((column//2), 2))
    n = 1
    i = 1
    transformed_points[0] = [points[0], points[1]]
    while i <= 7:
        transformed_points[n] = [points[i+1], points[i+2]]
        if i+2 == column - 1:
            break
        i+=2
        n+=1
    
    return transformed_points


def extract_points(boxes):
    # rearrange to get proper sizing
    row, column = boxes.shape
    points = np.empty((row, column*2))
    for i in range(0,row):
        points[i] = [boxes[i,0], boxes[i,1],
                     boxes[i,0] + boxes[i,2], boxes[i,1],
                     boxes[i,0] + boxes[i,2], boxes[i,1] + boxes[i,3],
                     boxes[i,0], boxes[i,1] + boxes[i,3]]
    
    
       
    # Now create shapes dictionary for transfer to labelme
    shapes = []
    for i in range(0,row):
        transformed_points = transform_points(points[i])
        shapes.append(
            dict(
                label="draftmarks",
                points=transformed_points, # x-y point of shape that is being drawn
                shape_type="polygon", #we can dictate more shapes here but this seems to be the most robust
                flags={}, #flags that we create
                group_id=None, # used for gouping similar items
                other_data=None, # not really sure.....
            )
        )
    return points, shapes
    

class myEncoder(json.JSONEncoder):
    # Handles the default behavior of
    # the encoder when it parses an object 'obj'
    def default(self, obj):
        # If the object is a numpy array
        if isinstance(obj, np.ndarray):
            # Convert to Python List
            return obj.tolist()
        else:
            # Let the base class Encoder handle the object
            return json.JSONEncoder.default(self, obj)


class LabelFile:
    
    #initializes the class
    def __init__(self, filename=None, shapes=[]):
        self.shapes = shapes
        self.imagePath = None
        self.imageData = None
        self.load(filename)
        self.filename = filename

    
    def load(self, filename):
        image_data = load_image_file(filename)
        image_path = osp.dirname(filename)+filename #this causes the directory of the image file plus the filename to be stored
        flags = {} #this portion would need updating based on flags we want to add
        self.imagePath = image_path
        self.imageData = image_data
        self.flags = flags
        
        
    
    #this part checks the height and width of image and stores it
    @staticmethod
    def _check_image_height_and_width(imageData):
        img_arr = utils.img_b64_to_arr(imageData)
        image_height = img_arr.shape[0]
        image_width = img_arr.shape[1]
        return image_height, image_width
    
    #this portion saves and writes the JSON file
    def save(
        self,
        filename,
        shapes,
        image_path,
        image_data,
        image_height=None,
        image_width=None,
        other_data=None,
        flags=None,
    ):
        if image_data is not None:
            image_data = base64.b64encode(image_data).decode("utf-8")
            image_height, image_width = self._check_image_height_and_width(image_data)
        if other_data is None:
            other_data = {}
        if flags is None:
            flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            imagePath=image_path,
            imageData=image_data,
            imageHeight=image_height,
            imageWidth=image_width,
        )
        if filename.endswith(".jpg"):
            filename = filename[:-4]
        with open(filename+".json", "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, cls=myEncoder)
            
