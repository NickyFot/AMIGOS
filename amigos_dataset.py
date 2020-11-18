from torch.utils import data

from .utils import *


class AMIGOS(data.Dataset):
    """Class to handle AMIGOS Dataset.

    A class to handle and get video data from the AMIGOS dataset.
        Typical usage example:
        amigos = AMIGOS(
        root_path=<path_to_dataset>,
        annotation_path='amigos.json',
        spatial_transform=transforms.ToTensor(),
        feature_type='RGB'
    )

    """
    def __init__(
            self,
            root_path: str,
            annotation_path: str,
            spatial_transform: callable = None,
            temporal_transform: callable = None,
            target_transform: callable = None,
            feature_type: str = None
    ):
        """
        Dataset constructor
        :param root_path: (str) path to content root
        :param annotation_path: (str) path to annotations file
        :param spatial_transform: (callable) transformation to apply to each frame
        :param temporal_transform: (callable) transformation to apply to clip
        :param target_transform: (callable) transformation to apply to target
        :param feature_type: (str) feature type to return Options: Meta, RGB
        :returns AMIGOS Dataset object
        """
        if feature_type not in FEATURES:
            raise NotImplemented('Feature type not implemented')
        self.data = self._make_dataset(
            root_path,
            annotation_path,
            file_extension='.jpg' if feature_type == FEATURES[-1] else '.json'
        )
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader(feature_type)
        self.labels = [x['label'] for x in self.data]
        self.indices = list(range(0, len(self.data)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (clip, target)
        """
        path = self.data[index]['video']
        clip = self.loader(path)
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0)
        if self.temporal_transform is not None:
            clip = self.temporal_transform(clip)
        target = self.data[index]['label']
        if self.target_transform is not None:
            target = self.target_transform(target)
        return clip, target

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _make_dataset(
            root_path: str,
            annotation_path: str,
            file_extension: str = '.json'
    ) -> list:
        dataset = load_annotation_data(annotation_path)
        video_names, annotations = get_video_names_and_annotations(dataset)
        filenames = get_file_names(root_path, file_extension)
        if file_extension == '.jpg':
            filenames = list(set([os.path.dirname(x) for x in filenames]))
        dataset = []
        for idx in range(len(filenames)):
            if file_extension != '.jpg':
                froot = os.path.dirname(filenames[idx])
            else:
                froot = filenames[idx]
            segment = os.path.basename(froot)
            subject_vid = os.path.basename(os.path.dirname(froot)).split('_')[:2]
            subject_vid.append(segment)
            key = '_'.join(subject_vid)
            video_path = filenames[idx]

            if key not in video_names:
                continue
            j = video_names.index(key)
            if not os.path.exists(video_path):
                continue
            if len(annotations[j]) == 0:
                continue
            sample = {
                'video': video_path,
                'video_id': video_names[j],
                'label': annotations[j]
            }
            dataset.append(sample)
        return dataset
