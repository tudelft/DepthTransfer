from isaacgym.torch_utils import *
import torch
from isaacgym.torch_utils import tensor_clamp

import numpy as np
import matplotlib.pyplot as plt

class TensorLowPassFilter:
    def __init__(self,cutoff_freqs,dt):
        self.dt = dt
        self.cutoff_freqs = cutoff_freqs
        self.smoothing_alpha = self.calc_smoothing_factor(self.cutoff_freqs,self.dt)
        print("smooting factors:", self.smoothing_alpha)
        self._prev_y = None

    @staticmethod
    def calc_smoothing_factor(cutoff_freqs,dt):
        tmp = 2*np.pi*dt*cutoff_freqs
        return (tmp) / (tmp + 1)

    def update_cutoff_freqs(self,cutoff_freqs):
        self.cutoff_freqs = cutoff_freqs
        self.smoothing_alpha = self.calc_smoothing_factor(self.cutoff_freqs,self.dt)

    def filter(self,x):
        if self._prev_y is None:
            self._prev_y = self.smoothing_alpha * x
            return self._prev_y
        else:
            self._prev_y = self.smoothing_alpha * x + (1 - self.smoothing_alpha) * self._prev_y
            return self._prev_y

    def get_latest(self):
        return self._prev_y

class TensorIMU:
    def __init__(self, num_envs, dt, 
                    config, gravity, device = torch.device("cuda")):
        self.num_envs = num_envs
        self.dt = dt
        self.cfg = config
        self.acc_std=self.cfg.acc_std
        self.acc_bias_std=self.cfg.acc_bias_std
        self.acc_bias_p=self.cfg.acc_bias_p
        self.gyro_std=self.cfg.gyro_std
        self.gyro_bias_std=self.cfg.gyro_bias_std
        self.gyro_bias_p=self.cfg.gyro_bias_p

        self.debug = self.cfg.debug
        self.device = device

        self.g_world = torch.tensor(gravity, device=self.device, requires_grad=False, dtype=torch.float32)

        if self.cfg.use_low_pass_filter:
            cutoff_freqs = torch.ones((1,6,1),device=self.device)
            cutoff_freqs[:3] *= self.cfg.low_pass_acc_cutoff_freq
            cutoff_freqs[3:] *= self.cfg.low_pass_gyro_cutoff_freq
            self.low_pass_filter = TensorLowPassFilter(cutoff_freqs, dt)

        # IMU frame to body frame, as well as orthogonality and diagonality compensation
        #self.R_imu # rotation matrix from imu frame to body frame
        #self.trans_imu # offset from imu frame to body frame
        #self.S = self.R_imu @ O @ D
        self.gyro_correction = torch.tensor(self.cfg.gyro_correction, device=self.device, dtype=torch.float32)
        self.acc_correction = torch.tensor(self.cfg.acc_correction, device=self.device, dtype=torch.float32)
        
        I = torch.eye(3, device = self.device, dtype=torch.float32)
        
        self.print_params()
        self.gyro_correction = self.gyro_correction.expand(self.num_envs, -1, -1)
        self.acc_correction = self.acc_correction.expand(self.num_envs, -1, -1)
        self.g_world = self.g_world.expand(self.num_envs, -1)
        # Continuous system
        self.A = torch.block_diag(-I*self.acc_bias_p,-I*self.gyro_bias_p)

        self.R = torch.block_diag(I*(self.acc_std**2),I*(self.gyro_std**2)) # self.acc_correction @ I*acc_std**2 @ self.acc_correction.T, self.gyro_correction @ I*gyro_std**2 @ self.gyro_correction.T) 
        self.Q = torch.block_diag(I*self.acc_bias_std**2,I*self.gyro_bias_std**2)

        self.C = self.C_d = torch.block_diag(I,I)
        self.D = self.D_d = torch.block_diag(I,I)
        
        G = torch.eye(6, device=self.device)

        GQGT = G @ self.Q @ G.T

        # Discretized system

        # build blockmat
        tmp_x,tmp_y = torch.cat((-self.A, GQGT,torch.zeros_like(self.A,device=self.device), self.A.T),dim=1).t().chunk(2)
        exponent = self.dt * torch.cat((tmp_x,tmp_y),dim=1).t()

        VanLoanMatrix = torch.linalg.matrix_exp(exponent)

        self.A_d = VanLoanMatrix[6:,6:].T
        self.GQGT_d = self.A_d @ VanLoanMatrix[:6, 6:]

        # exponent = dt * np.block([[-A_c, GQGT_c],
        #                       [np.zeros_like(A_c), A_c.T]])
        # VanLoanMatrix = scipy.linalg.expm(exponent)
        # A_d = VanLoanMatrix[15:, 15:].T
        # GQGT_d = A_d @ VanLoanMatrix[:15, 15:]


        self.R_d = 1/self.dt * self.R


        # if self.debug:

        #     print("Continuous system")
        #     print("A", self.A)
        #     print("C", self.C)
        #     print("D", self.D)
        #     print("Q", self.Q)
        #     print("GQGT", GQGT)
        #     print("R", self.R)

        #     print("Discretized system")
        #     print("A_d", self.A_d)
        #     print("C_d", self.C_d)
        #     print("D_d", self.D_d)
        #     print("GQGT_d", self.GQGT_d)
        #     print("R_d", self.R_d)

        self.std_mat = torch.block_diag(torch.sqrt(self.GQGT_d), torch.sqrt(self.R_d)) # not sure if correct

        self.x = torch.zeros((self.num_envs,6,1), device = self.device)
        # self.accel_bias = torch.zeros((num_envs, 3), device = self.device)
        # self.ang_rate_bias = torch.zeros((num_envs, 3), device = self.device)
    
    def print_params(self):
        print(vars(self))
    

    def calc_noises(self):
        nn = torch.randn((self.num_envs,12,1),device=self.device)
        mm = self.std_mat.expand(self.num_envs,-1,-1)
        noises = torch.bmm(mm, nn)
        return noises

    def meas_and_update(self, robot_quat, sensor_quat, accel_t, ang_rate_t, world_frame=False):

        '''
        world_frame: if accel_t and ang_rate_t are in world frame or not
        '''
        noises = self.calc_noises()
        bias_noises = noises[:,:6]
        
        accel_noise = noises[:,6:9]
        ang_noise = noises[:,9:12]

        accel_bias = self.x[:,:3]
        ang_rate_bias = self.x[:,3:]

        if world_frame:
            hm = quat_rotate(quat_mul(robot_quat, sensor_quat), (accel_t - self.g_world)).unqueeze(2)
            ang_rate = quat_rotate(quat_mul(robot_quat, sensor_quat), ang_rate_t).unsqueeze(2)
        else:
            hm = (accel_t - quat_rotate(quat_mul(robot_quat, sensor_quat), self.g_world)).unsqueeze(2)
            ang_rate = ang_rate_t.unsqueeze(2)

        accel_meas = hm + torch.bmm(self.acc_correction, (accel_bias + accel_noise))
        ang_rate_meas = ang_rate + torch.bmm(self.gyro_correction, ang_rate_bias + ang_noise)
        self.x = self.A_d @ self.x + bias_noises
        
        if self.cfg.use_low_pass_filter:
            
            filtered_meas = self.low_pass_filter.filter(torch.concat([accel_meas,ang_rate_meas],dim=1))
            accel_meas = filtered_meas[:,:3]
            ang_rate_meas = filtered_meas[:,3:]

        return accel_meas.squeeze(2), ang_rate_meas.squeeze(2)

    
    def reset(self):
        self.x.zero_()
    
    def reset_idx(self, env_ids):
        self.x[env_ids, :] = 0.


# if __name__ == "__main__":

#     device="cuda"

#     acc_corr = gyro_corr = torch.eye(3,device=device)

#     num_envs = 512

#     N = 1000
#     S = 4
#     g = 9.81

#     dt = 1/100

#     low_pass_filter_conf = {
#         'low_pass_acc_cutoff_freq' : 40.5, # Hz
#         'low_pass_gyro_cutoff_freq' : 39.9 # Hz, see BMI160 specs
#     }

#     imu = TensorIMU(
#     num_envs=num_envs,
#     dt=dt,
#     acc_std=1.167e-3,  # Accelerometer standard deviation, TUNABE
#     acc_bias_std=4e-3,  # Accelerometer bias standard deviation
#     acc_bias_p=1e-16,  # Accelerometer inv time constant see (10.57)

#     gyro_std=4.36e-5,  # Gyro standard deviation
#     gyro_bias_std=5e-5,  # Gyro bias standard deviation
#     gyro_bias_p=1e-16,  # Gyro inv time constant see (10.57)

#     acc_correction=acc_corr,  # Accelerometer correction matrix
#     gyro_correction=gyro_corr,  # Gyro correction matrix
#     g = g,
#     use_low_pass_filter=True,
#     low_pass_filter_kwargs=low_pass_filter_conf
#     )


#     accel_t = torch.zeros(N,num_envs,3,1,device=device)# + g
#     ang_rate_t = torch.zeros(N,num_envs,3,1,device=device)

#     for i in range(S):
#         a = np.random.random()*2.0 - 1.0
#         w = np.random.random()*0.5 - 0.25
#         accel_t[i*(N//S):(i+1)*(N//S)] = 0.1* a
#         ang_rate_t[i*(N//S):(i+1)*(N//S)] = 0.1*w


#     accel_meas = torch.zeros(N,num_envs,3,1,device=device)
#     ang_rate_meas = torch.zeros(N,num_envs,3,1,device=device)

#     accel_bias = torch.zeros(N,num_envs,3,1,device=device)
#     gyro_bias = torch.zeros(N,num_envs,3,1,device=device)


#     I = torch.eye(3,device=device)


#     rot_mats = I.expand(num_envs,-1,-1)

#     for i in range(N):
#         if i % 200 == 0:
#             print("resetting biases")
#             #imu.reset()
#             imu.reset_idx(range(num_envs))
#         biases = imu.x
#         accel_bias[i] = biases[:,:3]
#         gyro_bias[i] = biases[:,3:]
#         accel_meas[i], ang_rate_meas[i] = imu.meas_and_update(rot_mats, accel_t[i], ang_rate_t[i])


#     plt.subplot(411)
#     plt.plot(accel_meas.cpu()[:,0,0])
#     plt.plot(accel_t.cpu()[:,0,0])

#     plt.subplot(412)
#     plt.plot(ang_rate_meas.cpu()[:,0,0])
#     plt.plot(ang_rate_t.cpu()[:,0,0])

#     plt.subplot(413)
#     plt.plot(accel_bias.cpu()[:,0,0])
#     #plt.plot(accel_t.cpu()[:,0,0])

#     plt.subplot(414)
#     plt.plot(gyro_bias.cpu()[:,0,0])
#     #plt.plot(ang_rate_t.cpu()[:,0,0])


#     plt.show()
#     plt.savefig("imu-testing.png")
#     return 0


@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@torch.jit.script
def quat_axis(q, axis=0):
    # type: (Tensor, int) -> Tensor
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)