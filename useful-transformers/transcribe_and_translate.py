# import subprocess
# import json
# import requests
# import sys
# import os

# # 1. 语音转英文文本
# def transcribe_wav(wav_path, model="base.en", lang="en"):
#     cmd = [
#         "taskset", "-c", "4-7",
#         "./venv/bin/python", "-m", "useful_transformers.transcribe_wav",
#         wav_path, model, lang
#     ]
#     try:
#         output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
#         # 取最后一行非空行作为识别文本
#         lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
#         return lines[-1]
#     except subprocess.CalledProcessError as e:
#         print("识别失败：", e.output)
#         return ""

# # 2. 调用本地 DeepSeek Ollama API 翻译英文到中文
# # def translate_with_deepseek(english_text):
# #     url = "http://localhost:11434/api/generate"
# #     payload = {
# #         "model": "deepseek-r1:1.5b",
# #         "prompt": (
# #             "你是翻译助手。请直接将下列英文翻译成简洁、通顺的中文，不要解释，不要说明过程，"
# #             "不要输出任何与翻译无关的内容，只输出翻译后的中文：\n\n"
# #             f"{english_text}"
# #         ),
# #         "stream": False,
# #         "options": {
# #             "temperature": 0.2,
# #             "top_p": 0.9,
# #             "max_tokens": 128
# #         }
# #     }

# #     response = requests.post(url, json=payload)
# #     if response.status_code == 200:
# #         result = response.json()
# #         return result.get("response", "").strip(" \n\"'。，！？<>")
# #     else:
# #         return f"翻译失败，状态码: {response.status_code}"

# def translate_with_deepseek(english_text):
#     url = "http://localhost:11434/api/generate"
#     payload = {
#         "model": "deepseek-r1:1.5b",
#         "prompt": (
#             "你是翻译助手。请直接将下列英文翻译成简洁、通顺的中文，不要解释，不要说明过程，"
#             "不要输出任何与翻译无关的内容，只输出翻译后的中文：\n\n"
#             f"{english_text}"
#         ),
#         "stream": False,
#         "options": {
#             "temperature": 0.2,
#             "top_p": 0.9,
#             "max_tokens": 128
#         }
#     }

#     response = requests.post(url, json=payload)
#     if response.status_code == 200:
#         result = response.json().get("response", "")
#         # 去除思考内容（如果有）
#         if "</think>" in result:
#             result = result.split("</think>")[-1]
#         return result.strip(" \n\"'。，！？<>")
#     else:
#         return f"翻译失败，状态码: {response.status_code}"


# # 3. 主函数
# def main():
#     if len(sys.argv) != 2:
#         print("用法：python transcribe_and_translate.py <wav文件路径>")
#         return

#     wav_file = sys.argv[1]
#     if not os.path.exists(wav_file):
#         print(f"找不到文件：{wav_file}")
#         return

#     print("🔊 开始识别语音内容...")
#     en_text = transcribe_wav(wav_file)
#     print("识别结果（英文）：", en_text)

#     print("🌐 正在翻译...")
#     zh_text = translate_with_deepseek(en_text)
#     print("翻译结果（中文）：", zh_text)

# if __name__ == "__main__":
#     main()
import subprocess
import json
import requests
import sys
import os
import shutil

# 1. 语音转英文文本
def transcribe_wav(wav_path, model="base.en", lang="en"):
    cmd = [
        "taskset", "-c", "4-7",
        "./venv/bin/python", "-m", "useful_transformers.transcribe_wav",
        wav_path, model, lang
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        # 取最后一行非空行作为识别文本
        lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
        return lines[-1]
    except subprocess.CalledProcessError as e:
        print("识别失败：", e.output)
        return ""

# 2. 调用本地 DeepSeek Ollama API 翻译英文到中文
def translate_with_deepseek(english_text):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "deepseek-r1:1.5b",
        "prompt": (
            # "你是翻译助手。请直接将下列英文按照字面意思翻译成简洁、通顺的中文，不要解释，不要说明过程，"
            # "不要输出任何与翻译无关的内容，只输出翻译后的中文：\n\n"
            "你是一位专业的翻译助手，请将以下英文内容翻译成简体中文。\n"
                "要求：\n"
                "1. 输出时要严格按照输入文本的格式和行数\n"
                "2. 只按照原文翻译，不要引申出其他意思\n"
                "3. 只输出中文翻译，不要输出和添加其他任何解释或说明\n"
            f"{english_text}"
        ),
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 128
        }
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json().get("response", "")
        if "</think>" in result:
            result = result.split("</think>")[-1]
        return result.strip(" \n\"'。，！？<>")
    else:
        return f"翻译失败，状态码: {response.status_code}"

# 3. 检查是否为合法 WAV 文件
def is_valid_wav(filepath):
    try:
        output = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name,codec_type", "-of", "json", filepath],
            text=True
        )
        info = json.loads(output)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "audio" and stream.get("codec_name") == "pcm_s16le":
                return True
        return False
    except subprocess.CalledProcessError:
        return False

# 4. 转换为合法 WAV 格式（单声道 16kHz）
def convert_to_wav(input_path, output_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-f", "wav", output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 5. 主函数
def main():
    if len(sys.argv) != 2:
        print("用法：python transcribe_and_translate.py <音频文件路径>")
        return

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"找不到文件：{input_file}")
        return

    # 检查并转换为合法 WAV 格式
    if input_file.lower().endswith(".wav") and is_valid_wav(input_file):
        wav_file = input_file
    else:
        print("输入文件不是合法 WAV，将进行转换...")
        temp_wav = "/tmp/converted_audio.wav"
        convert_to_wav(input_file, temp_wav)
        if not os.path.exists(temp_wav):
            print("转换失败。")
            return
        wav_file = temp_wav

    print("🔊 开始识别语音内容...")
    en_text = transcribe_wav(wav_file)
    print("识别结果（英文）：", en_text)

    print("🌐 正在翻译...")
    zh_text = translate_with_deepseek(en_text)
    print("翻译结果（中文）：", zh_text)

if __name__ == "__main__":
    main()
