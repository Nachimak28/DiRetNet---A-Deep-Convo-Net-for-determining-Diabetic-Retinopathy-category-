from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.wsgi import WSGIServer

app = Flask(__name__)
size = 128
Model_path = 'C:/Users/nachiket/Desktop/SEM_8/BE_project/Codes/Inception_retrained.h5'

model = load_model(Model_path)
print('Model loaded. Start serving...')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size = (size,size))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    op = (model.predict(x))*100
    preds = np.round_(op,4)
    
    return preds

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # get fine from post request
        f = request.files['file']

        #save file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        predictions = [preds[0][0],preds[0][1],preds[0][2],preds[0][3]]
        #predictions = [75,10,10,3,2]
        categories = ['Zero', 'One', 'Three', 'Four']
        source = ColumnDataSource(data = dict(categories = categories, predictions = predictions))
        col=['blue', 'green', 'orange', 'red']

        p = figure(x_range = categories, plot_height = 500, title = 'DR Categories')
        p.vbar(x = 'categories', top = 'predictions',source = source, legend = 'categories', width = 0.9, line_color = 'white', fill_color = factor_cmap('categories', palette = col,factors = categories))
        p.xgrid.grid_line_color = None
        p.y_range.start = 0

        show(p)
        opstr0 = 'Category 0: ' + str(preds[0][0])+'%'
        opstr1 = ', Category 1: ' + str(preds[0][1])+'%'
        #opstr2 = ', Category 2: ' + str(preds[0][2])+'%'
        opstr3 = ', Category 3: ' + str(preds[0][2])+'%'
        opstr4 = ', Category 4: ' + str(preds[0][3])+'%'
        result = opstr0+opstr1+opstr3+opstr4
        return result
    return None


if __name__ == '__main__':
    #serve app with gevent

    http_server = WSGIServer(('', 5000),app)
    http_server.serve_forever()


