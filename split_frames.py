import os
import sys
import subprocess
import datetime

import typing


def probe_file(filename: str) -> float:
    cmnd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', filename]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(filename)
    out, err = p.communicate()
    return float(out)


def get_segments(duration: float) -> typing.List[tuple]:
    seg0: tuple = (0, 20)
    segments = [seg0]
    for step in range(5, int(duration), 20):
        if step+20 < duration:
            segments.append((step, step+20))
    segments.append((duration-20, duration))
    return segments


def seconds_to_strtime(seconds: float) -> str:
    return str(datetime.timedelta(seconds=seconds))


def main(dir_path, dst_dir_path, class_name):
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_class_path):
        os.makedirs(dst_class_path)

    for file_name in os.listdir(class_path):
        if '.mov' not in file_name:
            continue
        name, ext = os.path.splitext(file_name)
        video_len = probe_file(os.path.join(*[class_path, file_name]))
        segments = get_segments(video_len)
        video_file_path = os.path.join(class_path, file_name)
        video_path, ext = video_file_path.split('.')
        video_file_path = video_file_path.replace(ext, '264')
        if not os.path.isfile(video_file_path):
            transform = 'ffmpeg -i {0}.{1} -c copy {0}.264'.format(video_path, ext)
            subprocess.call(transform, shell=True)
        for idx in range(len(segments)):
            dst_directory_path = os.path.join(*[dst_class_path, name, str(idx+1)])
            if not os.path.exists(dst_directory_path):
                os.makedirs(dst_directory_path)
            try:
                if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
                    subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                    print('remove {}'.format(dst_directory_path))
                    os.makedirs(dst_directory_path)
                else:
                    continue
            except:
                print(dst_directory_path)
                continue
            segment_strt = seconds_to_strtime(segments[idx][0])
            segment_end = seconds_to_strtime(segments[idx][1])
            # -vsync 0 -hwaccel cuvid -c:v h264_cuvid -vf hwdownload,format=nv12 # gpu flags
            cmd = 'ffmpeg -vsync 0 -hwaccel cuvid -c:v h264_cuvid -accurate_seek -ss \"{}\" -to \"{}\" -i \"{}\" -vf hwdownload,format=nv12,hwupload  -q:v 5 -f image2 \"{}/image_%05d.jpg\"'.format(segment_strt, segment_end, video_file_path, dst_directory_path)
            print(cmd)
            subprocess.call(cmd, shell=True)
            print('\n')


# main('Data', 'Frames', 'Exp1_P17_face')
if __name__ == "__main__":
    dir_path = sys.argv[1]
    dst_dir_path = sys.argv[2]

    for root, dirs, files in os.walk(dir_path):
        for class_name in dirs:
            main(root, dst_dir_path, class_name)
