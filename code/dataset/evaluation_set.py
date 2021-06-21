from .gif_dataset import GIFDataset
import numpy as np
import imageio
import torchvision.transforms as T
import torch


class EvaluationSet(GIFDataset):
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, data_root, num_frames=32, crop_size=(224, 224)):
        super(EvaluationSet, self).__init__(data_root)
        self.num_frames = num_frames
        self.crop_size = crop_size

    def __getitem__(self, idx):
        gif_path = self.GIF_list[idx][0]
        label = self.GIF_list[idx][1]

        # Read all frames
        frames = []
        for i in imageio.get_reader(gif_path):
            # Process single channel GIF
            if len(i.shape) == 2:
                i = np.stack([i] * 3, axis=-1)

            frames.append(i[:, :, :3])

        frames = np.array(frames)

        num_frames = frames.shape[0]
        if num_frames >= self.num_frames:
            # Sample some frames throughout the sequence
            frames_idx = np.round(np.linspace(0, num_frames-1, self.num_frames)).astype(np.uint)
            frames = frames[frames_idx]
        else:
            # Pad with edge
            num_frames_delta = self.num_frames - num_frames
            frames = np.pad(frames, pad_width=[(num_frames_delta // 2, num_frames_delta - num_frames_delta // 2), (0, 0), (0, 0), (0, 0)], mode='edge')

        transform = T.Compose(
            [T.ToPILImage(),
             T.Resize(self.crop_size),
             T.ToTensor(),
             T.Normalize(self.MEAN, self.STD)])

        frames = torch.stack([transform(i) for i in frames])

        return frames, label

