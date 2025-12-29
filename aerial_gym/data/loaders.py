""" Some data loading utilities """
from bisect import bisect
from os import listdir, path
from os.path import join, isdir
from tqdm import tqdm
import torch
import torch.utils.data
import numpy as np
import rosbag
from cv_bridge import CvBridge
import cv2
import random
from aerial_gym.mav_baselines.torch.models.vae_320 import min_pool2d

class _RolloutDataset(torch.utils.data.Dataset): # pylint: disable=too-few-public-methods
    def __init__(self, root, transform, device, file_num=15, buffer_size=320, train=True): # pylint: disable=too-many-arguments
        self._transform = transform
        self.device = device
        self._files = [
            join(root, sd, ssd)
            for sd in listdir(root) if isdir(join(root, sd))
            for ssd in listdir(join(root, sd)) if ssd.endswith('.pth')
        ]
        # random.shuffle(self._files)
        # print(self._files)
        if train:
            self._files = self._files[0:file_num]
        else:
            self._files = self._files[-1:]

        self._buffer = None
        self._buffer_index = 0
        self._file_index = 0
        self._buffer_size = buffer_size
        self._cum_size = 0

    def load_next_buffer(self):
        """ Loads next buffer """
        seg_size = self._buffer_size
        self._buffer = []

        for file_index in range(self._file_index, len(self._files)+1):
            if self._file_index >= len(self._files):
                self._file_index = 0
                file_index = 0
            fname = self._files[file_index]
            print("Loading file: ", fname)
            data = torch.load(fname, map_location='cpu')
            # print("Data keys: ", data.observations.keys())
            data_count = data.obs_vae.shape[0]
            print("Data count: ", data_count)
            if self._buffer_index + seg_size < data_count:
                # print("self._buffer_index: ", self._buffer_index)
                # print("seg_size: ", seg_size)
                data_seg = data.obs_vae[self._buffer_index:(self._buffer_index + seg_size)] / 255.0
                self._buffer += [sub_data for sub_data in data_seg]
                self._buffer_index += seg_size
                seg_size = self._buffer_size
                self._cum_size += 1
                break
            else:
                # print("self._buffer_index111: ", self._buffer_index)
                data_seg= data.obs_vae[self._buffer_index:] / 255.0
                self._buffer += [sub_data for sub_data in data_seg]
                seg_size = self._buffer_size - (data_count - self._buffer_index)
                self._buffer_index = 0
                self._file_index += 1
            # print("self._file_index: ", self._file_index, " len(self._files): ", len(self._files))

    def __len__(self):
        if not self._cum_size:
            self.load_next_buffer()
        return len(self._buffer)

    def __getitem__(self, i):
        if self._transform:
            return self._transform(self._buffer[i])
        return self._buffer[i]

class _RolloutDatasetOld(torch.utils.data.Dataset): # pylint: disable=too-few-public-methods
    def __init__(self, root, transform, buffer_size=320, train=True): # pylint: disable=too-many-arguments
        self._transform = transform

        self._files = [
            join(root, sd, ssd)
            for sd in listdir(root) if isdir(join(root, sd))
            for ssd in listdir(join(root, sd)) if ssd.endswith('.npz')
        ]
        random.shuffle(self._files)
        # print(self._files)
        if train:
            self._files = self._files[0:-10]
        else:
            self._files = self._files[-10:]

        self._buffer = None
        self._buffer_index = 0
        self._file_index = 0
        self._buffer_size = buffer_size
        self._cum_size = 0

    def load_next_buffer(self):
        """ Loads next buffer """
        seg_size = self._buffer_size
        self._buffer = []
        if self._file_index >= len(self._files):
            self._file_index = 0
        for file_index in range(self._file_index, len(self._files)):
            fname = self._files[file_index]
            with np.load(fname, allow_pickle=True) as data:
                data_count = data['observations'].item()['image'].shape[0]
                print("AB Data count: ", data_count)
                # if self._buffer_index + seg_size < data_count:
                #     data_seg = data['observations'].item()['image'][self._buffer_index:(self._buffer_index + seg_size)]
                #     self._buffer += [sub_data for sub_data in data_seg]
                #     self._buffer_index += seg_size
                #     seg_size = self._buffer_size
                #     self._cum_size += 1
                #     break
                # else:
                #     data_seg= data['observations'].item()['image'][self._buffer_index:]
                #     self._buffer += [sub_data for sub_data in data_seg]
                #     seg_size = self._buffer_size - (data_count - self._buffer_index)
                #     self._buffer_index = 0
                #     self._file_index += 1
                if self._buffer_index + seg_size < data_count:
                    # print("self._buffer_index: ", self._buffer_index)
                    # print("seg_size: ", seg_size)
                    # data_seg = data.obs_vae[self._buffer_index:(self._buffer_index + seg_size)] / 255.0
                    data_seg = data['observations'].item()['image'][self._buffer_index:(self._buffer_index + seg_size)]
                    self._buffer += [sub_data for sub_data in data_seg]
                    self._buffer_index += seg_size
                    seg_size = self._buffer_size
                    self._cum_size += 1
                    break
                else:
                    # print("self._buffer_index111: ", self._buffer_index)
                    data_seg= data['observations'].item()['image'][self._buffer_index:]
                    self._buffer += [sub_data for sub_data in data_seg]
                    seg_size = self._buffer_size - (data_count - self._buffer_index)
                    self._buffer_index = 0
                    self._file_index += 1

    def __len__(self):
        if not self._cum_size:
            self.load_next_buffer()
        return len(self._buffer)

    def __getitem__(self, i):
        if self._transform:
            return self._transform(self._buffer[i].squeeze().astype(np.uint8))
        return self._buffer[i]

# class RosbagDataset(torch.utils.data.Dataset):
#     def __init__(self, rosbag_folder, image_topic, buffer_size=10, transform=None, train=True):

#         self.image_topic = image_topic
#         self.transform = transform
#         self.bag_paths = [
#             join(rosbag_folder, sd, ssd)
#             for sd in listdir(rosbag_folder) if isdir(join(rosbag_folder, sd))
#             for ssd in listdir(join(rosbag_folder, sd))]
#         if train:
#             self.bag_paths = self.bag_paths[:-5]
#         else:
#             self.bag_paths = self.bag_paths[-5:]

#         self.bridge = CvBridge()
#         self._cum_size = None
#         self._buffer = None
#         self._buffer_fnames = None
#         self._buffer_index = 0
#         self._buffer_size = buffer_size
#         # rospy.init_node('rosbag_dataset', anonymous=True)

#     def load_next_buffer(self):
#         """ Loads next buffer """
#         self._buffer_fnames = self.bag_paths[self._buffer_index:(self._buffer_index + self._buffer_size)]
#         self._buffer_index += self._buffer_size
#         self._buffer_index = self._buffer_index % len(self.bag_paths)
#         self._buffer = []
#         self._cum_size = [0]

#         # progress bar
#         pbar = tqdm(total=len(self._buffer_fnames),
#                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
#         pbar.set_description("Loading file buffer ...")

#         for f in self._buffer_fnames:
#             with pyrosbag.Bag(f, "r") as bag:
#                 current_data = []
#                 for _, msg, _ in bag.read_messages(topics=[self.image_topic]):
#                     try:
#                         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") / 1000.0
#                         cv_image = (np.minimum(cv_image, 12.0)) / 12.0 * 255.0
#                         shape = cv_image.shape
#                         cv_image = cv_image[:, int((shape[1]-shape[0])/2 - 1) : int((shape[1]+shape[0])/2 - 1)]
#                         dim = (256, 256)
#                         cv_image = cv2.resize(cv_image, dim, interpolation=cv2.INTER_AREA)
#                         # convert to uint8
#                         cv_image = cv_image.astype(np.uint8)
#                         if self.transform:
#                             cv_image = self.transform(cv_image)
#                         current_data += [cv_image]
#                     except Exception as e:
#                         print(f"Error extracting image: {str(e)}")
#                 self._cum_size += [self._cum_size[-1] + len(current_data)]
#                 self._buffer.append(current_data)
#             pbar.update(1)
#         pbar.close()
    
#     def __len__(self):
#         # to have a full sequence, you need self.seq_len + 1 elements, as
#         # you must produce both an seq_len obs and seq_len next_obs sequences
#         if not self._cum_size:
#             self.load_next_buffer()
#         return self._cum_size[-1]
                
#     def __getitem__(self, i):
#         # binary search through cum_size
#         file_index = bisect(self._cum_size, i) - 1
#         seq_index = i - self._cum_size[file_index]
#         data = self._buffer[file_index]
#         return self._get_data(data, seq_index)

#     def _get_data(self, data, seq_index):
        
#         return data[seq_index]
    
# class RosbagSequenceDataset(RosbagDataset):
#     def __getitem__(self, index):
#         return np.array(self._buffer[index]).squeeze()
    
#     def __len__(self):
#         return len(self._buffer)

class RosbagDataset(torch.utils.data.Dataset):
    def __init__(self, rosbag_folder, image_topic, buffer_size=10, transform=None, with_gray=True, train=True):

        self.image_topic = image_topic
        self.transform = transform
        self.with_gray = with_gray
        self.bag_paths = [
            join(rosbag_folder, sd, ssd)
            for sd in listdir(rosbag_folder) if isdir(join(rosbag_folder, sd))
            for ssd in listdir(join(rosbag_folder, sd))]
        if train:
            self.bag_paths = self.bag_paths[1:]
        else:
            self.bag_paths = self.bag_paths[:1]
            # print(self.bag_paths)
        print("Number of bags: ", len(self.bag_paths))
        self.bridge = CvBridge()
        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = buffer_size
        # rospy.init_node('rosbag_dataset', anonymous=True)

    def load_next_buffer(self):
        """ Loads next buffer """
        self._buffer_fnames = self.bag_paths[self._buffer_index:(self._buffer_index + self._buffer_size)]
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self.bag_paths)
        self._buffer = []
        self._cum_size = [0]

        # progress bar
        pbar = tqdm(total=len(self._buffer_fnames),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Loading file buffer ...")

        for f in self._buffer_fnames:
            with rosbag.Bag(f, "r") as bag:
                current_data = []
                data = {}
                t_depth_frame = None
                t_gray_frame = None
                for topic, msg, t in bag.read_messages(topics=self.image_topic):
                    try:
                        if topic == '/camera/depth/image_rect_raw':
                            t_depth_frame = t.to_sec()
                        else:
                            t_gray_frame = t.to_sec()
                        if topic == '/camera/depth/image_rect_raw':
                            t_depth_frame = t.to_sec()
                            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") / 1000.0
                            depth_image = (np.minimum(depth_image, 5.0)) / 5.0 * 255.0
                            dim = (240, 320)
                            depth_image = cv2.resize(depth_image, dim, interpolation=cv2.INTER_AREA)
                            depth_image = depth_image.astype(np.uint8)
                            if self.transform:
                                depth_image = self.transform(depth_image)
                            data['depth'] = depth_image
                        elif topic == '/camera/infra1/image_rect_raw' and self.with_gray:
                            gray_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                            dim = (240, 320)
                            gray_image = cv2.resize(gray_image, dim, interpolation=cv2.INTER_AREA)
                            if self.transform:
                                gray_image = self.transform(gray_image)
                            data['gray'] = gray_image
                        if t_depth_frame and t_gray_frame and np.abs(t_depth_frame - t_gray_frame) < 0.01:
                            t_depth_frame = None
                            t_gray_frame = None
                            current_data.append(data)
                            data = {}
                        elif t_depth_frame and not self.with_gray:
                            t_depth_frame = None
                            current_data.append(data)
                            data = {}

                    except Exception as e:
                        print(f"Error extracting image: {str(e)}")
                self._cum_size += [self._cum_size[-1] + len(current_data)]
                self._buffer.append(current_data)
            pbar.update(1)
        pbar.close()
    
    def __len__(self):
        # to have a full sequence, you need self.seq_len + 1 elements, as
        # you must produce both an seq_len obs and seq_len next_obs sequences
        if not self._cum_size:
            self.load_next_buffer()
        return self._cum_size[-1]
                
    def __getitem__(self, i):
        # binary search through cum_size
        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index]
        return self._get_data(data, seq_index)

    # def _data_per_sequence(self, data_length):
    #     return data_length

    def _get_data(self, data, seq_index):
        return data[seq_index]

class RolloutLSTMSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, root, device, max_pooling_depth=True, seq_num_batch=10, train=True):
        self._files = [
            join(root, sd, ssd)
            for sd in listdir(root) if isdir(join(root, sd))
            for ssd in listdir(join(root, sd)) if ssd.endswith('.pth')
        ]
        random.shuffle(self._files)
        if train:
            self._files = self._files[:-5]
        else:
            self._files = self._files[-5:]

        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = 1
        self.device = device
        self.seq_num_batch = seq_num_batch
        self.max_pooling_depth = max_pooling_depth
        self.seq_idx = 0

    def load_next_buffer(self):
        """ Loads next buffer """
        self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]

        self._buffer = []
        self._cum_size = 0

        for f in self._buffer_fnames:
            data = torch.load(f, map_location='cpu')
            seq_num = data.lstm_states[0][0].shape[1]
            seq_len = data.observations['image'].shape[0] // seq_num
            training_len = min(50, seq_len)
            for i in range(self.seq_idx, min(self.seq_idx + self.seq_num_batch, seq_num)):
                observations = {key: data.observations[key][i*seq_len:(i*seq_len+training_len)].to(self.device) for key in data.observations.keys()}
                observations['image'] = min_pool2d(observations['image'], kernel_size=7, stride=1, padding=3)
                lstm_states = (data.lstm_states[0][0][:, i, :].to(self.device), data.lstm_states[0][1][:, i, :].to(self.device))
                episode_starts = data.episode_starts[i*seq_len:(i*seq_len+training_len)].to(self.device)
                self._buffer.append((observations, lstm_states, episode_starts))
                self._cum_size += 1

            self.seq_idx += self.seq_num_batch
            if self.seq_idx >= seq_num:
                self.seq_idx = 0
                self._buffer_index += self._buffer_size
                self._buffer_index = self._buffer_index % len(self._files)
            break

    def __len__(self):
        if not self._cum_size:
            self.load_next_buffer()
        return len(self._buffer)
    
    def __getitem__(self, i):
        return self._buffer[i]
