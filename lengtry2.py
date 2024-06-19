# from flask import Flask, send_file
# import os
# from flask import request
# app = Flask(__name__)


# AUDIO_FOLDER = "/root/RVC01/RVC/out/instrument_夏日重现.wav.reformatted.wav_10-mixed.wav"

# @app.route('/download', methods=['GET'])
# def download_file():
#     # 构建音频文件的完整路径
#     audio_path = AUDIO_FOLDER
#     # 检查文件是否存在
#     if os.path.exists(audio_path):
#         # 如果文件存在，则返回文件
#         return send_file(audio_path, as_attachment=True)
#     else:
#         # 如果文件不存在，则返回错误
#         return 'File not found', 404

# if __name__ == '__main__':
#     app.run(debug=False)




# @app.route('/download', methods=['GET'])
# def download_file():
#     file_path = request.args.get('path', '')

#     # 构建音频文件的完整路径
#     audio_path = file_path
#     # 检查文件是否存在
#     if os.path.exists(audio_path):
#         # 如果文件存在，则返回文件
#         return send_file(audio_path, as_attachment=True)
#     else:
#         # 如果文件不存在，则返回错误
#         return 'File not found', 404

# if __name__ == '__main__':
#     app.run(debug=True)


# # 定义一个函数，用于处理文件下载
# @app.route('/<filepath>', methods=['GET'])
# def download_file(filepath):
#     file_path = filepath
#     if os.path.exists(file_path):    
#         return send_file(file_path, as_attachment=True)
#     else:
#         # 如果文件不存在，则返回错误
#         return 'File not found', 404


# if __name__ == '__main__':
#     app.run(debug=False)




from flask import Flask, send_file, request
import os
app = Flask(__name__)

# AUDIO_FOLDER = 'path/to/your/audio/folder'  # 这里是默认的音频文件夹路径

@app.route('/download', methods=['GET'])
def download_file():
    # 从查询参数中获取音频文件名
    audio_filename = request.args.get('filename')
    
    # 构建音频文件的完整路径
    audio_path = audio_filename
    
    # 检查文件是否存在
    if os.path.exists(audio_path):
        # 如果文件存在，则返回文件
        return send_file(audio_path, as_attachment=True)
    else:
        # 如果文件不存在，则返回错误
        return 'File not found', 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=False)
