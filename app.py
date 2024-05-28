import time

from flask import request, render_template, redirect, Flask
import dclp_tool
import json
import os
import shutil

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return redirect('/main_page')


@app.route('/main_page', methods=['GET', 'POST'])
def main_page():
    # 统计./static/temp文件夹下的文件夹数量
    path = './static/temp'
    n = []
    if os.path.exists(path):
        files = os.listdir(path)
        for file in files:
            if os.path.isdir(path + '/' + file):
                files = os.listdir(path + '/' + file)
                n.append(len([file for file in files if file.split('.')[-1] in dclp_tool.file_ext]))
    if request.method == 'POST':
        # 在./static/temp文件夹下创建一个新的文件夹
        path = path + '/' + str(len(n))
        os.mkdir(path)
        image = request.files.getlist('images')
        # 将这些图片保存到本地
        for i, img in enumerate(image):
            img.save(path + '/' + str(i) + '.' + img.filename.split('.')[-1])
        n.append(len(image))
        return render_template('main_page.html', images=[], diagnosis=[], case_num=n, time='')
    else:
        image_id = request.args.get('image_id')
        if image_id is not None:
            begin_time = time.time()
            result = dclp_tool.detect_and_classify(path + '/' + str(image_id))
            end_time = time.time()
            time_ = 'Time: ' + str(round(end_time - begin_time, 3))
            return render_template('main_page.html', images=result['images'], diagnosis=list(result['Diagnosis'].values())[0], case_num=n, time=time_)
        else:
            return render_template('main_page.html', images=[], diagnosis=[], case_num=n,time='')


@app.route('/app_data', methods=['POST'])
def app_data():
    # 清空file文件夹
    if os.path.exists('./file'):
        shutil.rmtree('./file')
    os.mkdir('./file')
    length = request.form['length']
    # 将这些图片保存到本地
    for i in range(int(length)):
        img = request.files['image' + str(i)]
        img.save('./file/' + img.filename)
    result = dclp_tool.detect_and_classify('./file')
    return json.dumps(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
