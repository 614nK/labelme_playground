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


def extract_points(boxes):
    points = np.empty(boxes.shape)
    for i in range(0,10):
        points[i] = [boxes[i,0], boxes[i,1], boxes[i,0] + boxes[i,2], boxes[i,1] + boxes[i,3]]
       
    # Now create shapes dictionary for transfer to labelme
    shapes = [
        dict(
            label="draftmarks",
            points=points, # x-y point of shape that is being drawn
            shape_type="polygon", #we can dictate more shapes here but this seems to be the most robust
            flags={}, #flags that we create
            group_id=None, # used for gouping similar items
            other_data=None, # not really sure.....
        )
    ]
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
    def __init__(self, filename=None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)
        self.filename = filename
    
    def load(self, filename):
        image_data = load_image_file(filename)
        imagePath = osp.dirname(filename)+filename
        flags = {} #this portion would need updating based on flags we want to add
        shapes = [
                dict(
                    label="waterline", # we would fill this with the different label strings
                    points=[[209,77],[228,81],[228,99],[210,99],[208,88]], # x-y point of shape that is being drawn
                    shape_type="polygon", #we can dictate more shapes here but this seems to be the most robust
                    flags={}, #flags that we create
                    group_id=None, # used for gouping similar items
                    other_data=None, # not really sure.....
                ) 
                #for s in data["shapes"]
            ]
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = image_data
        self.flags = flags
        
        
    
    #this part checks the height and width of image and stores it
    @staticmethod
    def _check_image_height_and_width(imageData):
        img_arr = utils.img_b64_to_arr(imageData)
        imageHeight = img_arr.shape[0]
        imageWidth = img_arr.shape[1]
        return imageHeight, imageWidth
    
    #this portion saves and writes the JSON file
    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageData,
        imageHeight=None,
        imageWidth=None,
        otherData=None,
        flags=None,
    ):
        if imageData is not None:
            imageData = base64.b64encode(imageData).decode("utf-8")
            imageHeight, imageWidth = self._check_image_height_and_width(imageData)
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        try:
            if filename.endswith(".jpg"):
                filename = filename[:-4]
            with open(filename+".json", "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            #self.filename = filename
        except Exception as e:
            raise LabelFileError(e)
            
