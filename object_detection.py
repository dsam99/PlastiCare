from imageai.Detection import ObjectDetection

# loading model here in background
detector = ObjectDetection()

# use fat yolo since tiny yolo sux
detector.setModelTypeAsYOLOv3()
path = "./models/yolo.h5"

detector.setModelPath(path)
detector.loadModel()