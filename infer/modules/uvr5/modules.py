import os
import traceback
import logging

logger = logging.getLogger(__name__)

import ffmpeg
import torch

from configs.config import Config
from infer.modules.uvr5.mdxnet import MDXNetDereverb
from infer.modules.uvr5.vr import AudioPre, AudioPreDeEcho

config = Config()


# 假设MDXNetDereverb和AudioPre等模块以及config变量已经在代码的其他部分定义好了


import os
import traceback
import torch

# 假设config和logger已经被定义在其他地方
# from some_module import config, logger

def uvr(model_name, inp_path, save_root_vocal, save_root_ins, agg, format0):
    try:
        # 确保保存路径存在
        if not os.path.exists(save_root_vocal):
            os.makedirs(save_root_vocal)
        if not os.path.exists(save_root_ins):
            os.makedirs(save_root_ins)

        # 加载模型
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15, config.device)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(os.getenv("weight_uvr5_root"), model_name + ".pth"),
                device=config.device,
                is_half=config.is_half,
            )

        # 检查音频格式并重格式化（如果需要）
        info = ffmpeg.probe(inp_path, cmd="ffprobe")
        need_reformat = (
            info["streams"][0]["channels"] != 2
            or info["streams"][0]["sample_rate"] != "44100"
        )
        if True:
            tmp_path = os.path.join(os.environ["TEMP"], f"{os.path.basename(inp_path)}.reformatted.wav")
            os.system(f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y')
            inp_path = tmp_path

        # 处理音频
        is_hp3 = "HP3" in model_name
        pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0, is_hp3=is_hp3)

        # 构建并返回处理后的音频文件的绝对路径
    
        vocal_path = os.path.join(save_root_vocal, f"instrument_{os.path.basename(inp_path).replace('.reformatted.wav', '.reformatted.wav_10')}.{format0}")
        ins_path = os.path.join(save_root_ins, f"vocal_{os.path.basename(inp_path).replace('.reformatted.wav', '.reformatted.wav_10')}.{format0}")

     


        return vocal_path, ins_path

    except Exception as e:
        traceback.print_exc()
        raise e
    
    finally:
        # 清理资源
        if model_name == "onnx_dereverb_By_FoxJoy":
            del pre_fun.pred.model
            del pre_fun.pred.model_
        else:
            del pre_fun.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
# 使用示例
# vocal_path, ins_path = uvr(
#     model_name="your_model_name",
#     inp_path="path_to_your_input_audio_file",
#     save_root_vocal="path_to_save_vocal",
#     save_root_ins="path_to_save_instrumental",
#     agg="aggregation_parameter",
#     format0="output_format"
# )










