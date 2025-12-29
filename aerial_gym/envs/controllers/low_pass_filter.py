import torch
import math
import matplotlib.pyplot as plt

# Modified LowPassFilter class with additional dimension support
class LowPassFilter:
    def __init__(self, num_envs, dims, cutoff_frequency, sampling_frequency, initial_value, device):
        """
        Initialize the low-pass filter with given cutoff frequency, sampling frequency, and initial value.
        num_envs defines the number of environments, dims defines the dimensionality within each environment.
        """
        self.num_envs = num_envs
        self.dims = dims
        self.device = device
        self.sampling_frequency = torch.ones(num_envs, dims, device=device, requires_grad=False) * sampling_frequency
        self.cutoff_frequency = torch.ones(num_envs, dims, device=device, requires_grad=False) * cutoff_frequency
        self.denumerator = self.init_den(self.cutoff_frequency, self.sampling_frequency)
        self.numerator = self.init_num(self.cutoff_frequency, self.sampling_frequency)
        self.initial_value = torch.ones(num_envs, dims, device=device, requires_grad=False) * initial_value
        self.input = torch.stack([self.initial_value, self.initial_value], dim=2).to(device)
        self.output = torch.stack([self.initial_value, self.initial_value], dim=2).to(device)

    @staticmethod
    def init_den(fc, fs):
        """
        Initialize the denumerator coefficients of the low-pass filter.
        """
        K = torch.tan(math.pi * fc / fs)
        poly = K**2 + math.sqrt(2.0) * K + 1.0

        denumerator = torch.zeros(fc.shape[0], fc.shape[1], 2, device=fc.device, requires_grad=False)
        denumerator[:, :, 0] = 2.0 * (K**2 - 1.0) / poly
        denumerator[:, :, 1] = (K**2 - math.sqrt(2.0) * K + 1.0) / poly

        return denumerator

    @staticmethod
    def init_num(fc, fs):
        """
        Initialize the numerator coefficients of the low-pass filter.
        """
        K = torch.tan(math.pi * fc / fs)
        poly = K**2 + math.sqrt(2.0) * K + 1.0

        numerator = torch.zeros(fc.shape[0], fc.shape[1], 2, device=fc.device, requires_grad=False)
        numerator[:, :, 0] = (K**2) / poly
        numerator[:, :, 1] = 2.0 * numerator[:, :, 0]

        return numerator

    def add(self, sample):
        """
        Add a new sample to the filter and compute the filtered output.
        """
        if sample.shape[0] != self.num_envs or sample.shape[1] != self.dims:
            raise ValueError(f"Input sample dimensions ({sample.shape}) must match (num_envs, dims) ({self.num_envs}, {self.dims}).")

        x2 = self.input[:, :, 1]
        self.input[:, :, 1] = self.input[:, :, 0]
        self.input[:, :, 0] = sample
        out = self.numerator[:, :, 0] * x2 + \
              (self.numerator * self.input - self.denumerator * self.output).sum(dim=2)
        self.output[:, :, 1] = self.output[:, :, 0]
        self.output[:, :, 0] = out

        return out

    def __call__(self, i=None):
        """
        Get the current output of the filter. If an index is provided, return the specific element.
        """
        if i is None:
            return self.output[:, :, 0]
        else:
            return self.output[i, :, 0]

    def derivative(self, i=None):
        """
        Compute the derivative of the output (rate of change).
        """
        deriv = self.sampling_frequency * (self.output[:, :, 0] - self.output[:, :, 1])
        if i is None:
            return deriv
        else:
            return deriv[i]

    def valid(self):
        """
        Check if all the filter parameters are finite and valid.
        """
        return torch.isfinite(self.sampling_frequency).all() and \
               torch.isfinite(self.denumerator).all() and \
               torch.isfinite(self.numerator).all() and \
               torch.isfinite(self.input).all() and \
               torch.isfinite(self.output).all()
    
    def reset(self, env_ids: torch.Tensor = None, initial_value = None):
        if env_ids is not None:
            self.initial_value[env_ids] = torch.ones(env_ids.shape[0], self.dims, device=self.device, requires_grad=False) * initial_value if initial_value is not None else self.initial_value[env_ids]
            self.input[env_ids] = torch.stack([self.initial_value[env_ids], self.initial_value[env_ids]], dim=2).to(self.device)
            self.output[env_ids] = torch.stack([self.initial_value[env_ids], self.initial_value[env_ids]], dim=2).to(self.device)

# # Example usage with a specified device (e.g., CPU or CUDA)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_envs = 3  # Number of environments
# dims = 2  # Dimensionality within each environment
# cutoff_frequency = 12.0  # Example cutoff frequency
# sampling_frequency = 300.0  # Example sampling frequency
# initial_value = torch.zeros(num_envs, dims, device=device)  # Initial values for each environment and dimension

# # Initialize the low-pass filter with the device
# lpf = LowPassFilter(num_envs, dims, cutoff_frequency, sampling_frequency, initial_value, device)

# # Generate a test sine wave and composite signal on the specified device
# t = torch.linspace(0, 1, steps=100, device=device)  # time vector
# sine_wave = torch.sin(2 * math.pi * 5 * t).unsqueeze(1).repeat(1, dims)  # 5Hz sine wave, repeated for dims
# high_freq_sine_wave = torch.sin(2 * math.pi * 20 * t).unsqueeze(1).repeat(1, dims)  # 20Hz sine wave, repeated for dims
# composite_signal = sine_wave + 0.5 * high_freq_sine_wave  # Combine signals

# # Expand composite signal to match the expected input shape (num_envs, dims)
# composite_signal_samples = composite_signal.unsqueeze(0).repeat(num_envs, 1, 1).transpose(0, 1)  # Shape: (100, num_envs, dims)

# # Apply the filter to the composite signal and collect the outputs
# filtered_outputs = []
# derivative_outputs = []
# for sample in composite_signal_samples:
#     filtered_output = lpf.add(sample)
#     filtered_outputs.append(filtered_output)
#     derivative_output = lpf.derivative()
#     derivative_outputs.append(derivative_output)

# # Convert list of outputs to tensors for easier analysis
# filtered_outputs = torch.stack(filtered_outputs).cpu()
# derivative_outputs = torch.stack(derivative_outputs).cpu()

# # Correcting the derivative computation for the original signal to ensure proper dimensions
# # Recompute the derivative of the original composite signal for comparison
# original_derivative = []
# for i in range(1, len(composite_signal)):
#     diff = (composite_signal[i] - composite_signal[i-1]) * sampling_frequency
#     original_derivative.append(diff)
# original_derivative = torch.stack(original_derivative).unsqueeze(0).repeat(num_envs, 1, 1).cpu()  # Adjust dimensions

# # Plot the comparison of original and filtered derivatives for the first environment and dimension
# plt.figure(figsize=(10, 6))
# plt.plot(t[1:].cpu().numpy(), original_derivative[0, :, 0].numpy(), label='Derivative of Original Composite Signal', linestyle='--')
# plt.plot(t[1:].cpu().numpy(), derivative_outputs[:-1, 0, 0].numpy(), label='Derivative of Filtered Output (Env 1, Dim 1)')
# plt.plot(t[1:].cpu().numpy(), derivative_outputs[:-1, 0, 1].numpy(), label='Derivative of Filtered Output (Env 1, Dim 2)')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title('Comparison of Derivatives: Original vs Filtered Output (12Hz Cutoff, 300Hz Sampling)')
# plt.legend()
# plt.grid(True)
# plt.show()