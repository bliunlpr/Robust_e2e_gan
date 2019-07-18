from torch.utils.data.sampler import Sampler
import numpy as np
from data.data_loader import SequentialDataset
from collections import defaultdict


class SequentialDatasetWithLength(SequentialDataset):
    def __init__(self, *args, **kwargs):
        """
        SpectrogramDataset that splits utterances into buckets based on their length.
        Bucketing is done via numpy's histogram method.
        Used by BucketingSampler to sample utterances from the same bin.
        """
        super(SequentialDatasetWithLength, self).__init__(*args, **kwargs)
        audio_lengths = [self.load_audio_feat_len(utt_id) for utt_id in self.spe_utt_ids]
        hist, bin_edges = np.histogram(audio_lengths, bins="auto")
        audio_samples_indices = np.digitize(audio_lengths, bins=bin_edges)
        self.bins_to_samples = defaultdict(list)
        self.bins_to_samples_list = []
        for idx, bin_id in enumerate(audio_samples_indices):
            self.bins_to_samples[bin_id].append(idx)
        for bin, sample_idx in self.bins_to_samples.items():
            ##np.random.shuffle(sample_idx)
            self.bins_to_samples_list.extend(sample_idx)
        print(audio_lengths, self.bins_to_samples, self.bins_to_samples_list)

class BucketingSampler(Sampler):
    def __init__(self, data_source):
        """
        Samples from a dataset that has been bucketed into bins of similar sized sequences to reduce
        memory overhead.
        :param data_source: The dataset to be sampled from
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        assert hasattr(self.data_source, 'bins_to_samples')

    def __iter__(self):
        for bin, sample_idx in self.data_source.bins_to_samples.items():
            np.random.shuffle(sample_idx)
            for s in sample_idx:
                yield s

    def __len__(self):
        return len(self.data_source)
        
    def shuffle(self, epoch):
        np.random.shuffle(self.data_source.bins_to_samples)
