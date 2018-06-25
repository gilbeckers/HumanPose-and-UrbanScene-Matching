import json
import os
import sys

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

sys.path.insert(0,"/openpose-master/MultiPersonMatching")
import matching
import glob
import base64

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

UPLOAD_FOLDER = '/home/jochen/server/processing'
DOWNLOAD_FOLDER = '/home/jochen/server/json/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def makeImageInput():
    return render_template('imagePost.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "no file sended"
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return "no filename"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return processfile(filename)
    return

@app.route('/findmatch', methods=['POST'])
def return_posematch():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "no file sended"
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return "no filename"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            id = request.form.get('id');
            return findmatch(filename,id)
    return
@app.route('/uploadPose', methods=['POST'])
def add_new_pose():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"id":-1})
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return jsonify({"id":-1})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            id =0
            for json in glob.iglob("/home/jochen/poses/*"):
                id = id+1
            filepath ="/home/jochen/poses/pose"+str(id)
            os.system("mkdir -p " +filepath)
            os.system("mkdir -p " +filepath+"/fotos")
            os.system("mkdir -p " +filepath+"/json")
            os.system("mkdir -p " +filepath+"/Processedfotos")
            filename = filename.rsplit('.')[1]
            filename = "0."+filename
            file.save(os.path.join(filepath+"/fotos", filename))
            os.system("cd /openpose-master/")
            os.system("./build/examples/openpose/openpose.bin -write_keypoint_json "+filepath+"/json -image_dir "+filepath+"/fotos -write_images "+filepath+"/Processedfotos -no_display")
            os.system("mv "+filepath+"/json/0_keypoints.json " +filepath+"/json/0.json ")
            return jsonify({"id":id})
    return jsonify({"id":-1})

@app.route('/getPoses', methods=['GET'])
def return_html_poses():
    return render_template('AllPoses.html')

@app.route('/getAllPoses', methods=['GET'])
def get_all_poses():
    data = []
    count = 0
    for picture in glob.iglob("/home/jochen/poses/pose*/fotos/0.*"):
        with open(picture, "rb") as image_file:
            dummy ={}
            dummy['naam'] = picture.rsplit('/fotos')[0].rsplit('/poses/')[1]

            dummy['foto']= base64.b64encode(image_file.read()).decode('utf-8')
            data.append(dummy)
    #print(data)
    return json.dumps(data)



def processfile (filename):
    os.system("./build/examples/openpose/openpose.bin -write_keypoint_json /home/jochen/server/json -image_dir /home/jochen/server/processing -write_images /home/jochen/server/calculated_poses/ -no_display")
    os.system("mv -v /home/jochen/server/processing/* /home/jochen/server/poses")
    filename = filename.rsplit('.')[0]
    filename = filename +'_keypoints.json'
    json_file = open(os.path.join(app.config['DOWNLOAD_FOLDER'], filename), "r")
    json_data = json_file.read()
    return json_data #jsonify(json_data)

def findmatch(filename, id):
    os.system("./build/examples/openpose/openpose.bin -write_keypoint_json /home/jochen/server/json -image_dir /home/jochen/server/processing -write_images /home/jochen/server/calculated_poses/ -no_display")
    os.system("mv -v /home/jochen/server/processing/* /home/jochen/server/fotos/")

    input_json = "/home/jochen/server/json/"+filename.rsplit('.')[0] +'_keypoints.json'
    model_json = "/home/jochen/poses/pose"+id+"/json/0.json"
    input_image_path = "/home/jochen/server/fotos/"+filename
    model_image_path = "/home/jochen/poses/pose"+id+"/fotos/0.jpg"
    #find matched
    result,US,MP,SP = matching.match(model_json, input_json, model_image_path, input_image_path)
    ismatch = False
    if result>70 and result <= 100:
        ismatch = True
    with open(input_json) as json_file:
        json_decoded = json.load(json_file)
        json_decoded['match'] = ismatch
        json_decoded['score'] = result
        json_decoded['US'] = US
        json_decoded['MP'] = MP
        json_decoded['SP'] = SP
        return jsonify(json_decoded)
    return json_data#jsonify(json_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
