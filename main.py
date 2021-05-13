# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:53:23 2021

@author: HP
"""

from utils import *
import pickle
from pathlib import Path
import fastai
import torch
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from fastai.vision.all import *
import matplotlib.pyplot as plt
from lovasz_losses import lovasz_hinge


UPLOAD_FOLDER = os.path.join('./static/img')
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model_path = "./export.pkl"



def symmetric_lovasz(outputs, targets):
        return (lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1 - targets)) / 2

def get_predictions(img_path, learn_inf):
    
    y = learn_inf.predict(img_path)
    preds_s = torch.sigmoid(tensor(y[0]))
    p = [(preds_s>0.5)]
    return p[0].squeeze()
    

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route("/",methods=['GET','POST'])
def home():
    filename=""
    output=""
    
    
    if request.method=='POST':
        
        
       
        file = request.files['img']
        # if user does not select file, browser also
        # submit an empty part without filename
        
        if file and allowed_file(file.filename):
            
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            learn_inf = load_learner('export.pkl')
            
            predictions = get_predictions(img_path, learn_inf)
            pred_arx = predictions.numpy()
            im = Image.fromarray(pred_arx)
            im = im.to_thumb(101,101)
            output="test"+str(random.random())+".png"
            im.save('./static/output/'+output)
            print("name",output)
            
           
            
       
   
    
    return render_template('index.html',filename=filename,output=output)
