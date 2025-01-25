
import os
import h5py
import librosa
import numpy as np

import torch
from torchvision import datasets, transforms
from torchvision import transforms
from torch.utils.data import Dataset
from GSC import google12_v2
'''
args.dataset_dir/
├── PSMNIST/
├── google_speech_command_2/
└── SSC/
'''
# organize and changed from https://github.com/ZhangShimin1/TC-LIF
def load_dataset(args):
    root_path = args.dataset_dir
    if args.task == 'SSC':  #download from https://zenkelab.org/datasets/
        args.time_window = 250
        T = args.time_window
        max_time = 1.4
        args.input_dim = 700 //args.n_bins
        args.output_dim = 35
        (x_train, y_train), (x_test, y_test) = getData(root_path, args.task)
        train_loader = SpikeIterator(x_train, y_train, args.batch_size, T, 700, max_time, n_bins = args.n_bins, shuffle=True, device = args.device)
        test_loader = SpikeIterator(x_test, y_test, args.batch_size, T, 700, max_time, n_bins = args.n_bins, shuffle=False, device = args.device)
    elif args.task == 'GSC':
        google12_v2(root_path)   #download
        args.time_window = 101
        args.input_dim = 40
        args.output_dim = 12
        train_dataset = GCommandLoader(root_path + '/google_speech_command_2/processed/train',
                                    window_size=.02, max_len=101)
        test_dataset = GCommandLoader(root_path + '/google_speech_command_2/processed/test',
                                    window_size=.02, max_len=101)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory='cpu', sampler=None)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=None, num_workers=args.num_workers, pin_memory='cpu', sampler=None)
        
    elif args.task == 'SMNIST' or args.task == 'PSMNIST':
        args.time_window = 784
        args.input_dim = 1
        args.output_dim = 10
        is_cuda = True
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if is_cuda else {}
        dataset_train = datasets.MNIST(root_path, train=True, download=False,
                                       transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root_path, train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        raise NotImplementedError
    
    return train_loader, test_loader

class SpikeIterator:
    def __init__(self, X, y, batch_size, nb_steps, nb_units, max_time, n_bins = 1,shuffle=True,device='cuda:0'):
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        # self.max_time = max_time
        self.shuffle = shuffle
        self.labels_ = np.array(y, dtype=np.float32)
        self.num_samples = len(self.labels_)
        self.number_of_batches = np.ceil(self.num_samples / self.batch_size)
        self.sample_index = np.arange(len(self.labels_))
        # compute discrete firing times
        self.firing_times = X['times']
        self.units_fired = X['units']
        self.time_bins = np.linspace(0, max_time, num=nb_steps)

        self.n_bins = n_bins

        self.device = device
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.sample_index)
        self.counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.number_of_batches)

    def __next__(self):
        if self.counter < self.number_of_batches:
            batch_index = self.sample_index[
                          self.batch_size * self.counter:min(self.batch_size * (self.counter + 1), self.num_samples)]
            coo = [[] for i in range(3)]
            for bc, idx in enumerate(batch_index):
                times = np.digitize(self.firing_times[idx], self.time_bins)
                units = self.units_fired[idx]
                batch = [bc for _ in range(len(times))]

                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            i = torch.LongTensor(coo).to(self.device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(self.device)

            X_batch = torch.sparse.FloatTensor(i, v, torch.Size(
                [len(batch_index), self.nb_steps, self.nb_units])).to_dense().to(
                self.device)
            
            ###############################################################
            binned_len = X_batch.shape[-1]//self.n_bins
            binned_frames = torch.zeros((len(batch_index), self.nb_steps, binned_len)).to(
                self.device)
            for i in range(binned_len):
                binned_frames[:,:,i] = X_batch[:, :,self.n_bins*i : self.n_bins*(i+1)].sum(axis=-1)
            ###############################################################
            y_batch = torch.tensor(self.labels_[batch_index], device=self.device).long()

            X_batch = binned_frames
            self.counter += 1
            return X_batch.to(device=self.device), y_batch.to(device=self.device)

        else:
            raise StopIteration

def getData(root, dataset):
    dataset = dataset
    root_path = root + '/' + dataset
    train_file = h5py.File(os.path.join(root_path, dataset.lower()+'_train.h5'), 'r')
    test_file = h5py.File(os.path.join(root_path, dataset.lower()+'_test.h5'), 'r')

    x_train = train_file['spikes']
    y_train = train_file['labels']
    x_test = test_file['spikes']
    y_test = test_file['labels']
    return (x_train, y_train), (x_test, y_test)

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

GSCmdV2Categs = {
            'unknown': 0,
            'silence': 1,
            '_unknown_': 0,
            '_silence_': 1,
            '_background_noise_': 1,
            'yes': 2,
            'no': 3,
            'up': 4,
            'down': 5,
            'left': 6,
            'right': 7,
            'on': 8,
            'off': 9,
            'stop': 10,
            'go': 11}

def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx.get(target, 0))
                    spects.append(item)
    return spects

def spect_loader(path, window_size, window_stride, window, normalize, max_len=101):
    y, sr = librosa.load(path, sr=None)
    librosa_melspec = librosa.feature.melspectrogram(y, sr=sr, n_fft=480,
                                                     hop_length=160, power=1.0,
                                                     n_mels=40, fmin=40.0, fmax=4000)
    spect = librosa.power_to_db(librosa_melspec, ref=np.max)

    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)
    # print(spect.shape)
    # spect_list = [spect]
    # for k in range(1, 3):
    #     spect_list.append(librosa.feature.delta(spect, order=k))

    return spect

class GCommandLoader(Dataset):

    """A google command data set loader

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=126):
        class_to_idx = GSCmdV2Categs
        spects = make_dataset(root, class_to_idx)
        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        spect = spect.squeeze(0).permute(1, 0)#(length, dim)
        return spect, target

    def __len__(self):
        return len(self.spects)

