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

    def _get_decision(self, system_status, gpu_status, batch_size, num_workers):
        gpu_util = gpu_status['util']
        cpu_util = system_status['cpu_util']
        
        # this is a simple heuristic, you may need to adapt it according to your needs
        if gpu_util > 80 and cpu_util < 20:
            # decrease batch_size, increase num_workers
            return {'batch_size': max(1, batch_size - 1), 'num_workers': num_workers + 1}
        elif gpu_util < 20 and cpu_util > 80:
            # increase batch_size, decrease num_workers
            return {'batch_size': batch_size + 1, 'num_workers': max(1, num_workers - 1)}
        else:
            # return current configuration if no changes are needed
            return {'batch_size': batch_size, 'num_workers': num_workers}
        
    def cal_gradient(self, system_status, gpu_status, batch_size, num_workers):
        pass
