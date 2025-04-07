import yaml
from yaml.loader import SafeLoader

import util.util as util

class Options():
    def __init__(self):
        self.initialized = False
        
        with open('options/base_options.yaml') as f:    
            self.opt = yaml.load(f, Loader=SafeLoader)
        
    def initialize(self, exp=None):
        for key in self.opt.keys():
            setattr(self, key, self.opt[key])
            
        self.initialized = True

    def save_options(self):
        if self.initialized:
            out_file = 'checks/'+self.cgwas_dir+'options.yaml'
            vars_list = vars(self)
            del vars_list['opt']
            with open(out_file, 'w') as f:
                yaml.dump(vars_list, f)