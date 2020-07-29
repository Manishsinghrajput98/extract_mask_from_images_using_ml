import os
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
import sys
import argparse
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("-image", "--input", required=True)
args = vars(ap.parse_args())

image = args["input"]

class ManishXYZ123():
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513

  def __init__(self):
    self.graph = tf.Graph()
    graph_def = None
    graph_def = tf.GraphDef.FromString(open("model/frozen_inference_graph.pb", "rb").read()) 
    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')
    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')
    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

def drawSegment(inputFilePath):
  MODEL = ManishXYZ123()	
  jpeg_str = open(inputFilePath, "rb").read()
  orignal_im = Image.open(BytesIO(jpeg_str))	
  resized_im, seg_map = MODEL.run(orignal_im)
  random_name = str(uuid.uuid4())
  print("------------Image Processing-----------------")
  width, height = resized_im.size
  dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
  for x in range(width):
            for y in range(height):
                color = seg_map[y,x]
                (r,g,b) = resized_im.getpixel((x,y))
                if color == 0:
                    dummyImg[y,x,3] = 0
                else :
                    dummyImg[y,x] = [r,g,b,255]
  img = Image.fromarray(dummyImg)
  img.save('output/' + random_name + ".png")
drawSegment(image)

