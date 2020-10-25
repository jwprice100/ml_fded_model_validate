import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time


"""This is a work around for RTX cards. Not sure why it's necessary, 
   but the internet said to try it and it made CUDNN work for 
   convolutional algorithms.
"""
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

#General Keras Libraries
from tensorflow.keras.models import Sequential, Model, model_from_json

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Generate synthetic signal log-mag data used to train a model.')
  parser.add_argument('data_file', action="store", type=str, help="Input HDF5 data file containing fft data and detection data to visualize.")
  parser.add_argument('model_file', action="store", type=str, help="Path to model.")
  parser.add_argument('weights_file', action="store", type=str, help="Path to weights.")
  parser.add_argument('--start_frame', type=int, default=0, help='First frame to visualize.')
  parser.add_argument('--num_frames', type=int, default=1, help='Number of frames to visualize.')
  parser.add_argument('--time_between_frames', type=float, default=0.5, help='Time between frames during visualization.')
  args = parser.parse_args()  
  

  #data_file = "/media/james/Bulk Storage/data/5000_frames_128_fft.hdf5"
  ###Deal with Configuration###
  start_frame = args.start_frame
  num_frames = args.num_frames
  end_frame = start_frame+num_frames
  time_between_frames = args.time_between_frames
  

  detection_threshold = 0.5
  draw_guard = 3

  ###Load Data###
  print("Loading data...")
  f = h5py.File(args.data_file, 'r')   
  fft_data = f['fft_data'][start_frame:end_frame,:]
  det_data = f['detection_data'][start_frame:end_frame,:]


  fft_size = fft_data.shape[1]
  fft_mean = f['fft_mean']
  fft_std = f['fft_std']

  ###Load Model###
  print("Loading model...")
  model_file = open(args.model_file, "r")
  model_json = model_file.read()
  model_file.close()
  model = model_from_json(model_json)
  model.load_weights(args.weights_file)  

  ###Visualization###
  plt.ion()
  fig = plt.figure()
  ax = fig.add_subplot(111)  

  for current_frame in range(num_frames):
    current_fft_data = fft_data[current_frame, :]
    current_truth_det_data = det_data[current_frame]
    max_dets = int(det_data.shape[1]/3)
    
    #Normalize input data
    current_fft_data_norm = (current_fft_data - fft_mean) / fft_std
    container = np.zeros(shape=(1, fft_size, 1))
    container[0,:,0] = current_fft_data_norm

    #Make prediction
    predicted_det_data_norm = model.predict(container)[0]
    
    #Denormalize prediction data
    #print(f"Prediction Norm: {predicted_det_data_norm}")
    predicted_det_data = predicted_det_data_norm
    predicted_det_data[1::3] = predicted_det_data[1::3]*(fft_size/2)+(fft_size/2)
    predicted_det_data[2::3] = predicted_det_data[2::3]*fft_size
    
    #Plot FFT data
    ax.plot(current_fft_data, "blue")
    mean_y = np.mean(current_fft_data)
    
    num_false_alarms = 0
    missed_detections = 0

    for i in range(max_dets):
      #If there is a detection here, draw a bounding box around it
      truth_det = current_truth_det_data[3*i] == 1
      pred_det = predicted_det_data[3*i] > detection_threshold

      #Determine missed detections and false alarms
      if truth_det and not pred_det:
        missed_detections = missed_detections+1

      if pred_det and not truth_det:
        num_false_alarms = num_false_alarms+1

      ax.text(0.8, 0.9, f"False Alarms: {num_false_alarms}\nMissed Dets: {missed_detections}", size=12, transform = ax.transAxes)

      if pred_det:
        truth_center = current_truth_det_data[i*3+1]
        truth_bandwidth = current_truth_det_data[i*3+2]        
        
        pred_center = predicted_det_data[3*i+1]
        pred_bandwidth = predicted_det_data[3*i+2]

        #Determine center freq and bandwidth error
        center_error = np.abs(pred_center-truth_center)
        bandwidth_error = np.abs(pred_bandwidth-truth_bandwidth)

        #Draw a bounding box around the prediction data
        x_left = pred_center - pred_bandwidth/2 - draw_guard
        x_left_floor = int(np.floor(x_left))
        x_right = pred_center + pred_bandwidth/2 + draw_guard
        x_right_ceil = int(np.ceil(np.ceil(x_right)))

        #Sometimes the model just predicts weird values.
        if x_left_floor < 0:
          x_left_floor = 0

        if x_right_ceil < 0:
          x_right_ceil = x_left_floor + 1

        max_y = np.max(current_fft_data[x_left_floor:x_right_ceil])
        ax.plot([x_left, x_left], [mean_y, max_y], 'r')
        ax.plot([x_right, x_right], [mean_y, max_y], 'r')
        ax.plot([x_right, x_right], [mean_y, max_y], 'r')        

        #Draw a nice box showing how well the model did.
        label_height = current_fft_data[int(np.round(truth_center))]        
        label_string = "Predicted: {0:.2f}\nCenter Error: {1:.2f}\nBandwidth Error: {2:.2f}".format(pred_center, center_error, bandwidth_error)
        ax.text(truth_center, label_height, label_string, color='black', bbox={'facecolor':'red', 'alpha': 0.75, 'pad':5})

    plt.show()
    plt.pause(time_between_frames)
    ax.clear()
      





