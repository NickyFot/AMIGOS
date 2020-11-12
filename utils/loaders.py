import os
from PIL import Image
import numpy as np
import functools
import json


FEATURES = ['AUs', 'PE', 'RGB']


def pil_loader(path: str):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def json_loader(video_dir_path: str, file_loader: callable) -> list:
    raw_video = file_loader(open(video_dir_path))['features']
    frame_indices = len(raw_video)
    frames = [raw_video.get(str(key)) for key in range(frame_indices)]
    return frames


def feature_loader(video_dir_path: str, frame_indices: list, file_loader: callable) -> list:
    raw_video = file_loader(video_dir_path)
    frame_indices = [idx for idx in frame_indices if idx < len(raw_video)]
    frames = raw_video[frame_indices]
    return frames


def video_loader(video_dir_path: str, image_loader: callable) -> list:
    video = []
    frame_indices = len(os.listdir(video_dir_path))
    for i in range(1, frame_indices+1):
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video


def get_loader(data_type: str) -> callable:
    if data_type == FEATURES[-1]:
        return functools.partial(video_loader, image_loader=pil_loader)
    else:
        return functools.partial(json_loader, file_loader=json.load)


def load_annotation_data(data_file_path: str) -> dict:
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_video_names_and_annotations(data_dict: dict, subset: str = None) -> tuple:
    videonames_lst = list()
    annotations_lst = list()
    for pidx, videos in data_dict.items():
        # TODO: see how to handle subset
        # if subset:
        #     if not data_dict[pidx]['subset'] == subset:
        #         continue
        for sti_idx, segments in videos['Videos'].items():
            for seg_idx, annotation in segments['Segments'].items():
                video_key = "P{p_idx}_{sti_idx}_{seg_idx}".format(p_idx=pidx, sti_idx=sti_idx, seg_idx=seg_idx)
                videonames_lst.append(video_key)
                arousal = annotation['Arousal']
                valence = annotation['Valence']
                annotations_lst.append(np.asarray([arousal, valence]))
    return videonames_lst, annotations_lst


def get_file_names(path: str, file_extension: str) -> list:
    filenames = list()
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith(file_extension):
                filenames.append(os.path.join(root, filename))
    return filenames