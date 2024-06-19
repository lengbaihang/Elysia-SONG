import subprocess
import threading
import multiprocessing

# 使用 multiprocessing 模块
p1 = multiprocessing.Process(target=lambda: subprocess.Popen(["python", "inferleng-web0608.py"]))
p2 = multiprocessing.Process(target=lambda: subprocess.Popen(["python", "lengtry2.py"]))

p1.start()
p2.start()

p1.join()
p2.join()



# # 启动第一个Python脚本
# subprocess.Popen(["python", "/root/RVC01/RVC/inferleng-web0608.py"])

# # 启动第二个Python脚本
# subprocess.Popen(["python", "/root/RVC01/RVC/lengtry2.py"])


