import glob, os
import moviepy.editor as mp
import librosa
import numpy as np
import cv2


def generate_mel_spectrogram(video_path, n_mels=128, fmax=8000, target_size=(256, 256)):
    # 提取音频
    audio_path = 'extracted_audio.wav'
    video = mp.VideoFileClip(video_path)
    #video.audio.write_audiofile(audio_path, verbose=False, logger=None)

    # 加载音频文件
    y, sr = librosa.load(audio_path)

    # 生成MEL频谱图
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

    # 将频谱图转换为dB单位
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 归一化到0-255之间
    S_dB_normalized = cv2.normalize(S_dB, None, 0, 255, cv2.NORM_MINMAX)

    # 将浮点数转换为无符号8位整型
    S_dB_normalized = S_dB_normalized.astype(np.uint8)

    # 缩放到目标大小
    img_resized = cv2.resize(S_dB_normalized, target_size, interpolation=cv2.INTER_LINEAR)

    return img_resized


def data_processing(input_path, output_dir):
    # Define paths

    num = 0
    for video_path in glob.glob(input_path):
        #output_image_path = os.path.join(output_dir, video_path.split('/')[-1][:-4] + '.jpg')
        file_name = os.path.basename(video_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        image_name = file_name_without_ext + '.jpg'
        output_image_path = os.path.join(output_dir, image_name)

        if not os.path.exists(output_image_path):
            try:
                mel_spectrogram_image = generate_mel_spectrogram(video_path)
                cv2.imwrite(output_image_path, mel_spectrogram_image)
            except Exception as e:
                num += 1
                print(f"Error processing {video_path}: {e}")
    print(num)
