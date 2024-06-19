from pydub import AudioSegment

# 加载伴奏和人声音频文件
# 请确保文件路径正确，并且文件格式被pydub支持
accompaniment = AudioSegment.from_file("/root/RVC01/RVC/opt1/vocal_vivy02.wav.reformatted.wav_10.wav")
vocals = AudioSegment.from_file("/root/RVC01/RVC/opt/instrument_vivy02.wav.reformatted.wav_10.wav.wav")

# 调整两个音频的音量，确保它们混合后的效果听起来均衡
# 可以根据实际情况调整db值
accompaniment = accompaniment.apply_gain(3)  # 减少伴奏音量
vocals = vocals.apply_gain(0)                 # 增加人声音量

# 混合伴奏和人声
mixed_audio = accompaniment.overlay(vocals)

# 导出混合后的音频到文件
mixed_audio.export("/root/RVC01/RVC/hecheng/vivy02.wav", format="wav")
