import requests
import json
import re
import subprocess
import os
import time
import sys

class OCRTranslator:
    def __init__(self, ocr_path, models, configs):
        """
        初始化OCR翻译器
        :param ocr_path: OCR可执行文件路径
        :param models: 模型文件列表 [det_model, cls_model, rec_model]
        :param configs: 配置文件列表 [det_config, cls_config, rec_config]
        """
        self.ocr_path = ocr_path
        self.models = models
        self.configs = configs
        self.ollama_url = "http://localhost:11434/api/generate"
        
    def run_ocr(self, image_path):
        """
        运行OCR识别图片中的文字
        :param image_path: 图片路径
        :return: 识别出的文字列表
        """
        # 构建OCR命令
        cmd = [
            self.ocr_path,
            self.models[0], self.configs[0],
            self.models[1], self.configs[1],
            self.models[2], self.configs[2],
            image_path,
            "1"  # 设备ID
        ]
        
        # 执行OCR命令并捕获输出
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"OCR执行错误: {result.stderr}")
            return []
        
        # 解析OCR输出
        return self.parse_ocr_output(result.stdout)
    
    def parse_ocr_output(self, output):
        """
        解析OCR输出，提取识别到的文本
        :param output: OCR程序输出
        :return: 文本列表
        """
        texts = []
        # 使用正则表达式匹配识别结果
        pattern = r"rec text:([^\n]*)rec score"
        matches = re.findall(pattern, output)
        
        for match in matches:
            text = match.strip()
            if text:  # 过滤空文本
                texts.append(text)
        
        return texts
    
    def translate_with_deepseek(self, english_texts):
        """
        使用本地部署的DeepSeek 1.5B模型翻译文本
        :param english_texts: 需要翻译的文本列表
        :return: 翻译结果
        """
        if not english_texts:
            return "未识别到文本"
        
        # 合并文本为段落
        english_paragraph = "\n".join(english_texts)
        
           # 修复缩进问题
        payload = {
            "model": "deepseek-r1:1.5b",
            "prompt": (
                "你是一位专业的翻译助手，请将以下英文内容翻译成简体中文。\n"
                "要求：\n"
                "1. 输出时要严格按照输入文本的格式和行数\n"
                "2. 只输出'原始的英文:和对应的中文翻译以及简短的中文解释'这种形式，不要输出和添加其他任何解释或说明\n"
                # "你是一位专业的翻译助手，请将以下英文菜单内容翻译成简体中文。\n"
                # "要求：\n"
                # "1. 先将其转换成可能会出现在菜单上的英文单词，再翻译为简体中文\n"
                # "2. 只输出中文翻译结果，不要添加任何解释或说明\n"
                # "3. 保持原始文本的格式和行数\n"
                # "4. 如果识别不出英文单词，将其作为样子相近的单词进行翻译\n"
                # "5. 价格和货币符号保持原样\n\n"
                "需要翻译的内容：\n"
                f"{english_paragraph}"
            ),
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 2000
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 提取翻译结果并清理
            translation = result.get("response", "")
            if "</think>" in translation:
                translation = translation.split("</think>")[-1]
            return translation.strip(" \n\"'。，！？<>")
            
        except Exception as e:
            print(f"翻译请求失败: {e}")
            return f"翻译服务错误: {str(e)}"
    
    def process_image(self, image_path):
        """
        处理图片：OCR识别 + 翻译
        :param image_path: 图片路径
        :return: 翻译结果
        """
        print(f"开始处理图片: {image_path}")
        
        # 步骤1: OCR识别
        start_time = time.time()
        ocr_texts = self.run_ocr(image_path)
        ocr_time = time.time() - start_time
        
        if not ocr_texts:
            return "未识别到文本"
        
        print(f"OCR识别结果 ({len(ocr_texts)} 条, 耗时 {ocr_time:.2f}秒):")
        for i, text in enumerate(ocr_texts, 1):
            print(f"{i}. {text}")
        
        # 步骤2: 翻译
        start_time = time.time()
        translation = self.translate_with_deepseek(ocr_texts)
        trans_time = time.time() - start_time
        
        print(f"\n翻译结果 (耗时 {trans_time:.2f}秒):")
        return translation

if __name__ == "__main__":
    # 检查参数
    if len(sys.argv) < 2:
        print("用法: python ocr_translate.py <图片路径>")
        sys.exit(1)
    
    # 配置参数 - 根据你的实际路径修改
    OCR_EXE = "./demo"  # OCR可执行文件路径
    MODELS = [
        "rk3588_det.rknn",   # 检测模型
        "rk3588_cls.rknn",   # 分类模型
        "rk3588_rec.rknn"    # 识别模型
    ]
    CONFIGS = [
        "det.yaml",          # 检测配置
        "cls.yaml",          # 分类配置
        "rec.yaml"           # 识别配置
    ]
    
    # 创建OCR翻译器
    translator = OCRTranslator(OCR_EXE, MODELS, CONFIGS)
    
    # 处理图片
    IMAGE_PATH = sys.argv[1]
    if not os.path.exists(IMAGE_PATH):
        print(f"错误: 文件不存在 - {IMAGE_PATH}")
        sys.exit(1)
        
    result = translator.process_image(IMAGE_PATH)
    
    # 打印最终结果
    print("\n" + "=" * 50)
    print("最终翻译结果:")
    print("=" * 50)
    print(result)
    
    # 保存结果到文件
    output_file = f"{os.path.splitext(IMAGE_PATH)[0]}_translation.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\n翻译结果已保存到: {output_file}")