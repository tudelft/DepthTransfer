
from collections import deque
import os

import torch.distributed as dist
import numpy as np
import torch


#@torch.jit.script
class TensorDeque: 
    """
    Tensor deque
    NB: using python deque is alot faster for many operations, but for our episode logger this is better
    """
    def __init__(self, maxlen: int, device: str):
        self.device = device
        self.maxlen = maxlen
        self.dq = torch.empty(self.maxlen,dtype=torch.float32,device=self.device)
        self.i = 0

    @torch.jit.export
    def append(self, x: torch.Tensor):
        if len(x.shape) == 0:
            x = x[None] # convert 0-d tensor

        x_len = min(len(x),self.maxlen) # cannot append more than x_len last elements of x

        if self.i < self.maxlen:
            avail_spots = self.maxlen - self.i # how many unfilled spots left in deque
            to_remove = max(x_len-avail_spots,0) # how many elements to remove from deque if we reach max capacity

            self.dq = torch.cat([self.dq[to_remove:self.i],x[-x_len:]])
            self.i = min(self.i + x_len, self.maxlen)
        else:
            self.dq = torch.cat([self.dq[x_len:],x[-x_len:]])

    @torch.jit.unused
    def __len__(self) -> int:
        return self.i

    @torch.jit.unused
    def __repr__(self) -> str:
        return self.dq[:self.i].__repr__()

    @torch.jit.unused
    def get(self):
        return self.dq[:self.i]


class TensorEpisodeLogger:
    def __init__(self, logger_episode_length=1001, ep_length_multiplier=2, device="cuda:0", multi_gpu=False):
        self.crash_count_dq = TensorDeque(logger_episode_length,device)
        self.spawn_crash_count_dq = TensorDeque(logger_episode_length,device)
        self.success_count_dq = TensorDeque(logger_episode_length,device)
        self.timeout_count_dq = TensorDeque(logger_episode_length,device)

        self.metric_deques = {
            "episode_length": TensorDeque(logger_episode_length*ep_length_multiplier,device),
            "crash_episode_length": TensorDeque(logger_episode_length*ep_length_multiplier,device),
            "alive_reset_distance": TensorDeque(logger_episode_length*ep_length_multiplier,device),
            "dead_reset_distance": TensorDeque(logger_episode_length*ep_length_multiplier,device),
        }

        self.logger_episode_length = logger_episode_length
        self.ep_length_multiplier = ep_length_multiplier

        self.multi_gpu = multi_gpu
        self.device = device

        if self.multi_gpu:
            self.rank = int(os.getenv("LOCAL_RANK", "0"))
            self.rank_size = int(os.getenv("WORLD_SIZE", "1"))
            self.statistics_tensor = torch.zeros(5+len(self.metric_deques)*2,device=self.device)
            self.statistics_tensor_joined = torch.zeros((self.rank_size,5+len(self.metric_deques)*2), dtype=torch.float32, device=self.device)

            # dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)

        print("init tensor logger")

    def update_lists(self, success_count, timeout_count, crash_count, spawn_crash_count, episode_lengths, alive_reset_distances, dead_reset_distances, crash_episode_lengths):
        self.crash_count_dq.append(crash_count)
        self.spawn_crash_count_dq.append(spawn_crash_count)
        self.success_count_dq.append(success_count)
        self.timeout_count_dq.append(timeout_count)
        self.metric_deques["episode_length"].append(episode_lengths)
        self.metric_deques["crash_episode_length"].append(crash_episode_lengths)
        self.metric_deques["alive_reset_distance"].append(alive_reset_distances)
        self.metric_deques["dead_reset_distance"].append(dead_reset_distances)

    def calc_combined_metrics(self, lengths, means, stds):
        # https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
        
        n = lengths[0]
        mean_x = means[0]
        std_x = stds[0]
        for i in range(1,len(lengths)):

            m = lengths[i]
            mean_y = means[i]
            std_y = stds[i]

            mean_new = (mean_x * n + mean_y * m) / (m+n)
            std_new = torch.sqrt(((n-1)*std_x**2 + (m-1)*std_y**2)/(n+m-1) + n*m*(mean_x-mean_y)**2/((n+m)*(n+m-1)))

            n = n + m
            mean_x = mean_new
            std_x = std_new

        return n, mean_x, std_x

    def get_stats(self, need_stats=False):
        if self.multi_gpu:
            
            need_stats_tensor = torch.tensor([1.0 * need_stats],device=self.device)
            dist.all_reduce(need_stats_tensor, op=dist.ReduceOp.SUM)
            if need_stats_tensor[0] == 0: # nobody needs stats:
                return None

            #print(f"rank: {self.rank} getting stats")

            success_sum = torch.sum(self.success_count_dq.get())
            timeout_sum = torch.sum(self.timeout_count_dq.get())
            crash_sum = torch.sum(self.crash_count_dq.get())
            spawn_crash_sum = torch.sum(self.spawn_crash_count_dq.get())
            history_count = torch.tensor(len(self.success_count_dq.get()),device=self.device)

            metric_list = [success_sum, timeout_sum, crash_sum, spawn_crash_sum, history_count]

            metric_stats = {}
            for metric_name, metric_dq in self.metric_deques.items():
                metric_stats[metric_name] = {}
                metric_stats[metric_name]["length"] = torch.tensor(len(metric_dq.get()),device=self.device)
                metric_stats[metric_name]["mean"] = torch.mean(metric_dq.get())
                metric_stats[metric_name]["std"] = torch.std(metric_dq.get())

                metric_list.extend([metric_stats[metric_name]["length"], metric_stats[metric_name]["mean"], metric_stats[metric_name]["std"]])

            self.statistics_tensor[:] = torch.tensor(metric_list,device=self.device)

            dist.all_gather_into_tensor(self.statistics_tensor_joined, self.statistics_tensor)
            sum_tensor = torch.sum(self.statistics_tensor_joined[:,:5],dim=0)
            
            metric_stats['success_sum'] = sum_tensor[0].item()
            metric_stats['timeout_sum'] = sum_tensor[1].item()
            metric_stats['crash_sum'] = sum_tensor[2].item()
            metric_stats['spawn_crash_sum'] = sum_tensor[3].item()
            metric_stats['history_count'] = sum_tensor[4].item()

            # if self.rank == 0:
            #     print(self.statistics_tensor_joined)
            #     print(sum_tensor)

            for i, metric_name in enumerate(self.metric_deques.keys()):
                N, mean, std = self.calc_combined_metrics(self.statistics_tensor_joined[:,5+3*i],self.statistics_tensor_joined[:,5+3*i],self.statistics_tensor_joined[:,6+3*i])
                metric_stats[metric_name]["N"] = N.item()
                metric_stats[metric_name]["mean"] = mean.item()
                metric_stats[metric_name]["std"] = std.item()

            return metric_stats

        else:

            metric_stats = {}
            metric_stats['success_sum'] = torch.sum(self.success_count_dq.get()).item()
            metric_stats['timeout_sum'] = torch.sum(self.timeout_count_dq.get()).item()
            metric_stats['crash_sum'] = torch.sum(self.crash_count_dq.get()).item()
            metric_stats['spawn_crash_sum'] = torch.sum(self.spawn_crash_count_dq.get()).item()
            metric_stats['history_count'] = len(self.success_count_dq.get())

            for metric_name, metric_dq in self.metric_deques.items():
                metric_stats[metric_name] = {}
                metric_stats[metric_name]["mean"] = torch.mean(metric_dq.get()).item()
                metric_stats[metric_name]["std"] = torch.std(metric_dq.get()).item()

            return metric_stats


class EpisodeLogger:
    def __init__(self, logger_episode_length=1001, ep_length_multiplier=4):
        self.crash_count_list = deque(maxlen=logger_episode_length)
        self.success_count_list = deque(maxlen=logger_episode_length)
        self.timeout_count_list = deque(maxlen=logger_episode_length)
        self.episode_lengths_list = deque(maxlen=logger_episode_length*ep_length_multiplier)
        self.alive_reset_distance_list = deque(maxlen=logger_episode_length*ep_length_multiplier)
        self.crash_timings_list = deque(maxlen=logger_episode_length*ep_length_multiplier)
        self.spawn_crash_count_list = deque(maxlen=logger_episode_length)

        self.logger_episode_length = logger_episode_length
        self.ep_length_multiplier = ep_length_multiplier

    def update_lists(self, success_count=0, timeout_count=0, crash_count=0, spawn_crash_count=0, episode_lengths_list=[], alive_reset_distance=[], crash_timings_list=[]):
        self.crash_count_list.append(crash_count)
        self.success_count_list.append(success_count)
        self.timeout_count_list.append(timeout_count)
        self.episode_lengths_list.extend(episode_lengths_list)
        self.alive_reset_distance_list.extend(alive_reset_distance)
        self.spawn_crash_count_list.append(spawn_crash_count)
        self.crash_timings_list.append(crash_timings_list)

    def get_stats(self):
        success_sum = np.sum(self.success_count_list)
        timeout_sum = np.sum(self.timeout_count_list)
        crash_sum = np.sum(self.crash_count_list)
        spawn_crash_sum = np.sum(self.spawn_crash_count_list)
        total_sum = success_sum + timeout_sum + crash_sum - spawn_crash_sum
        episode_lengths_mean = np.mean(self.episode_lengths_list) if len(self.episode_lengths_list) > 0 else 0.0
        episode_lengths_std = np.std(self.episode_lengths_list) if len(self.episode_lengths_list) > 0 else 0.0
        alive_reset_distance_mean = np.mean(self.alive_reset_distance_list) if len(self.alive_reset_distance_list) > 0 else 0.0
        alive_reset_distance_std = np.std(self.alive_reset_distance_list) if len(self.alive_reset_distance_list) > 0 else 0.0
        crash_timings_mean = np.mean(self.crash_timings_list) if len(self.crash_timings_list) > 0 else 0.0
        crash_timings_std = np.std(self.crash_timings_list) if len(self.crash_timings_list) > 0 else 0.0
        # make a dict here
        self.logger_stats = {}
        self.logger_stats["success_sum"] = success_sum
        self.logger_stats["timeout_sum"] = timeout_sum
        self.logger_stats["crash_sum"] = crash_sum - spawn_crash_sum
        self.logger_stats["episode_lengths_mean"] = episode_lengths_mean
        self.logger_stats["episode_lengths_std"] = episode_lengths_std
        self.logger_stats["alive_reset_distance_mean"] = alive_reset_distance_mean
        self.logger_stats["alive_reset_distance_std"] = alive_reset_distance_std
        self.logger_stats["num_episodes"] = len(self.success_count_list)
        self.logger_stats["success_rate"] = success_sum / total_sum if total_sum > 0 else 0.0
        self.logger_stats["timeout_rate"] = timeout_sum / total_sum if total_sum > 0 else 0.0
        self.logger_stats["crash_rate"] = (crash_sum - spawn_crash_sum) / total_sum if total_sum > 0 else 0.0
        self.logger_stats["spawn_crash_sum"] = spawn_crash_sum
        self.logger_stats["crash_timings_mean"] = crash_timings_mean
        self.logger_stats["crash_timings_std"] = crash_timings_std
        
        return self.logger_stats
        # return success_sum, timeout_sum, crash_sum, episode_lengths_mean, episode_lengths_std, alive_reset_distance_mean, alive_reset_distance_std, len(self.success_count_list)

