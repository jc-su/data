class PolicyInterface:
    def __init__(self, system_status, gpu_status, new_parameters, old_parameters):
        self.system_status = system_status
        self.gpu_status = gpu_status
        self.parameters = new_parameters["batch_size"], new_parameters["num_workers"]
        self.old_parameters = old_parameters["batch_size"], old_parameters["num_workers"]

    def decide(self):
        return self._get_decision(self.system_status, self.gpu_status, self.parameters, self.old_parameters)
    

    def detect_bolleneck(self, system_status, gpu_status):
        pass
    
    def _get_decision(self, system_status, gpu_status, parameters, old_parameters):
        pass


class PrototypePolicy(PolicyInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        pass
