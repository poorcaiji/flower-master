 # -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import os
import traceback
import threading
import uuid
from werkzeug.utils import secure_filename
from forecast import predict_flower, _load_model_and_classes

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['BATCH_FOLDER'] = 'batch_uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 确保上传目录存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 确保批量上传目录存在
if not os.path.exists(app.config['BATCH_FOLDER']):
    os.makedirs(app.config['BATCH_FOLDER'])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 预先加载模型（在后台线程中）
def preload_model():
    print("预加载模型中...")
    _load_model_and_classes()
    print("模型预加载完成")

# 启动预加载
threading.Thread(target=preload_model).start()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dp')
def t1_page():
    return render_template('dp.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件被上传'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 直接调用预测函数
            try:
                results = predict_flower(filepath)
                
                return jsonify({
                    'success': True,
                    'filepath': filepath,
                    'results': results
                })
            except Exception as e:
                error_details = traceback.format_exc()
                return jsonify({'error': f'执行识别脚本错误: {str(e)}', 'details': error_details}), 500
        
        return jsonify({'error': '不支持的文件类型'}), 400
    
    except Exception as e:
        error_details = traceback.format_exc()
        return jsonify({'error': f'服务器错误: {str(e)}', 'details': error_details}), 500

@app.route('/batch-upload', methods=['POST'])
def batch_upload():
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': '没有文件被上传'}), 400
        
        files = request.files.getlist('files[]')
        
        if not files or len(files) == 0:
            return jsonify({'error': '没有选择文件'}), 400
        
        # 创建批量处理的唯一文件夹
        batch_id = str(uuid.uuid4())
        batch_folder = os.path.join(app.config['BATCH_FOLDER'], batch_id)
        os.makedirs(batch_folder, exist_ok=True)
        
        batch_results = []
        
        for file in files:
            if file.filename == '':
                continue
                
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(batch_folder, filename)
                file.save(filepath)
                
                try:
                    # 调用预测函数
                    results = predict_flower(filepath)
                    
                    # 只保留最高置信度的结果
                    top_result = results[0] if results else None
                    
                    batch_results.append({
                        'filename': filename,
                        'filepath': filepath,
                        'result': top_result
                    })
                except Exception as e:
                    batch_results.append({
                        'filename': filename,
                        'filepath': filepath,
                        'error': str(e)
                    })
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'results': batch_results
        })
    
    except Exception as e:
        error_details = traceback.format_exc()
        return jsonify({'error': f'批量处理错误: {str(e)}', 'details': error_details}), 500

if __name__ == '__main__':
    app.run(debug=True)
    app.run(debug=True)