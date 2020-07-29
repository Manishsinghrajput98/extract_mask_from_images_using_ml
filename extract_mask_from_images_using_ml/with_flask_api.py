import os
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
import sys
import argparse
from flask import Flask, request, redirect, jsonify
from pykakasi import kakasi
import uuid
import cv2

class Manishxyz123(Flask):
    def __init__(self, host, name,sess):
        super(Manishxyz123, self).__init__(name,static_url_path='')
        self.host = host
        self.define_uri()
        self.requests = {}
        self.sess = sess
        self.INPUT_TENSOR_NAME = 'ImageTensor:0'
        self.OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
        self.INPUT_SIZE = 513

    def define_uri(self):
        self.provide_automatic_option = False
        self.add_url_rule('/start', None, self.start,methods=['POST'])

    def setup_converter(self):
        mykakasi = kakasi()
        mykakasi.setMode('H', 'a')
        mykakasi.setMode('K', 'a')
        mykakasi.setMode('J', 'a')
        self.converter = mykakasi.getConverter()

    def myrun(self, image):
    	width, height = image.size
    	resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    	target_size = (int(resize_ratio * width), int(resize_ratio * height))
    	resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    	batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME,feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    	seg_map = batch_seg_map[0]
    	return resized_image, seg_map

    def display_instances(self,image_path):
    	jpeg_str = open(image_path, "rb").read()
    	orignal_im = Image.open(BytesIO(jpeg_str))
    	resized_im, seg_map = self.myrun(orignal_im)
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
    	img.save('output-flask-api/' + random_name + ".png")

    def start(self):
        print(("json :",request.get_json()))
        if request.method == 'POST':
            body = request.get_json()
            print(("body :: {}".format(body)))
            image_path = body['image_path']
            self.display_instances(image_path)        
            res = dict()
            res['status'] = '200'
            res['result'] = "background-reemoved"
            print('====background-reemoved====',res)
            return jsonify(res)
            	
def importargs():
    parser = argparse.ArgumentParser('This is a server of Deeplabs')
    parser.add_argument("--host", "-H", help = "host name running server",type=str, required=False, default='localhost')
    parser.add_argument("--port", "-P", help = "port of runnning server", type=int, required=False, default=8080)
    args = parser.parse_args()
    return args.host, args.port

def main():
    host, port  = importargs()
    graph = tf.Graph()
    graph_def = None
    graph_def = tf.GraphDef.FromString(open("model/frozen_inference_graph.pb", "rb").read()) 
    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')
    with graph.as_default():
      tf.import_graph_def(graph_def, name='')
    sess = tf.Session(graph=graph)
    server = Manishxyz123(host, 'Deeplabs-model',sess,)
    server.run(host=host, port=port)
if __name__ == "__main__":
    main()