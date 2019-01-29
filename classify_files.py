import tensorflow as tf
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from shutil import copy

mypath = sys.argv[1]

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(len(onlyfiles))

# holt labels aus file in array 
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("tf_files/retrained_labels.txt")]
# !! labels befinden sich jeweils in eigenen lines -> keine aenderung in retrain.py noetig -> falsche darstellung im windows editor !!
				   
# graph einlesen, wurde in train.sh -> call retrain.py trainiert
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
 
    graph_def = tf.GraphDef()	## The graph-graph_def is a saved copy of a TensorFlow graph; objektinitialisierung
    graph_def.ParseFromString(f.read())	#Parse serialized protocol buffer data into variable
    _ = tf.import_graph_def(graph_def, name='')	# import a serialized TensorFlow GraphDef protocol buffer, extract objects in the GraphDef as tf.Tensor
	
	#https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/inception.py ; ab zeile 276

for file_id in onlyfiles:
	image_path = mypath + file_id
	print(image_path)
	image_data = tf.gfile.FastGFile(image_path, 'rb').read()		

	with tf.Session() as sess:

		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
		# return: Tensor("final_result:0", shape=(?, 4), dtype=float32); stringname definiert in retrain.py, zeile 1064 

		predictions = sess.run(softmax_tensor, \
				 {'DecodeJpeg/contents:0': image_data})
		# gibt prediction values in array zuerueck:
		
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
		# sortierung; circle -> 0, plus -> 1, square -> 2, triangle -> 3; array return bsp [3 1 2 0] -> sortiert nach groesster uebereinstimmmung

		# output
		for node_id in top_k:
			human_string = label_lines[node_id]
			score = predictions[0][node_id]
			if human_string == "include":
				include_score = score
			elif human_string == "exclude":
				exclude_score = score
			print('%s (score = %.5f)' % (human_string, score))
			
	
	if include_score >= 0.9:
		dst = os.getcwd() + "/include_90_100"
	elif include_score >= 0.8:
		dst = os.getcwd() + "/include_80_90"
	elif include_score >= 0.7:
		dst = os.getcwd() + "/include_70_80"
	elif include_score >= 0.6:
		dst = os.getcwd() + "/include_60_70"
	elif include_score >= 0.5:
		dst = os.getcwd() + "/include_50_60"
	elif include_score >= 0.4:
		dst = os.getcwd() + "/exclude_50_60"
	elif include_score >= 0.3:
		dst = os.getcwd() + "/exclude_60_70"
	elif include_score >= 0.2:
		dst = os.getcwd() + "/exclude_70_80"
	elif include_score >= 0.1:
		dst = os.getcwd() + "/exclude_80_90"
	elif include_score < 0.1:
		dst = os.getcwd() + "/exclude_90_100"


	copy(image_path, dst)
		
