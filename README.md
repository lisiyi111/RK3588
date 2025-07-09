#需要先部署deepseek1.5b
#进行图像识别并翻译
cd ./rk3588_ocr/workspace python3 ocr_translate.py <图片路径>
#进行语音识别并翻译
cd ./useful-transformers python3 transcribe_and_translate.py <音频路径>
