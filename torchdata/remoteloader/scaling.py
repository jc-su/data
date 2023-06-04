import numpy as np


class PolicyInterface:
    def __init__(self, system_status, gpu_status, batch_size, num_workers):
        self.system_status = system_status
        self.gpu_status = gpu_status

    def decide(self):
        return self._get_decision(self.system_status, self.gpu_status, self.batch_size, self.num_workers)

    def detect_bolleneck(self, system_status, gpu_status):
        # detrmine if the system is bottlenecked by CPU or GPU
        pass

    def _get_decision(self, system_status, gpu_status, parameters, old_parameters):
        pass


class PrototypePolicy(PolicyInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _cal_gradient(self, system_status, gpu_status):
        cpu_util_gradient = [i for i in np.gradient(system_status['cpu_util']) if i < 0]
        gpu_util_gradient = [i for i in np.gradient(gpu_status['util']) if i < 0]
        average_cpu_util_gradient = sum(cpu_util_gradient) / len(cpu_util_gradient)
        average_gpu_util_gradient = sum(gpu_util_gradient) / len(gpu_util_gradient)
        return average_cpu_util_gradient, average_gpu_util_gradient

    def detect_bolleneck(self, system_status, gpu_status, threshold=0.1):
        average_cpu_util_gradient, average_gpu_util_gradient = self._cal_gradient(system_status, gpu_status)
        if average_cpu_util_gradient < threshold and average_gpu_util_gradient > threshold:
            return "GPU"
        elif average_cpu_util_gradient > threshold and average_gpu_util_gradient < threshold:
            return "CPU"
        else:
            return "None"

    def _get_decision(self, system_status, gpu_status, current_parameters):
        # heuristic
        # if the system is bottlenecked by CPU, increase the number of workers
        # if the system is bottlenecked by GPU, increase the batch size
        new_parameters = current_parameters
        bottleneck = self.detect_bolleneck(system_status, gpu_status)
        if bottleneck == "CPU":
            # increase the number of workers
            new_parameters['num_workers'] = current_parameters['num_workers'] * 2
        elif bottleneck == "GPU":
            # increase the batch size
            new_parameters['batch_size'] = current_parameters['batch_size'] * 2
        elif bottleneck == "None":
            pass
        else:
            raise Exception("Unknown bottleneck")

        return new_parameters
