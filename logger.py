import os
import json
import datetime
import pandas as pd
import numpy as np

class LightLogger:
    """
    
    """
    
    def __init__(self, result_dir='result_logs', log_date=None):
        if log_date is None:
            log_date = str(datetime.date.today())
        self.result_dir = result_dir
        self.log_dir = f'{result_dir}/{log_date}'
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.configs = {}
        self.mem = {}
        self.current_log = None
        self.logging_vars = set()
    
    def set_configs(self, configs):
        self.configs = configs
    
    def start(self, note=None):
        if note is None:
            note = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + str(np.random.choice(100000))
        self.current_log = note
        self.mem[note] = {'time': note}
        for k in self.configs.keys():
            self.mem[note][k] = self.configs[k]
        self.mem[note]['current_step'] = -1
        self.mem[note]['log'] = {}
        return note
        
    
    def log(self, log_dict, step=None):
        if step is None:
            step = self.mem[self.current_log]['current_step'] + 1
        self.mem[self.current_log]['current_step'] = step
        self.mem[self.current_log]['log'][step] = {}
        for k in log_dict.keys():
            self.mem[self.current_log]['log'][step][k] = log_dict[k]
            self.logging_vars.add(k)
            
    def finish(self, save_now=True):
        if save_now:
            with open(f'{self.log_dir}/{self.current_log}.json', 'w') as f:
                json.dump(self.mem[self.current_log], f)
        self.current_log = None
    
    def to_df(self, tidy=True):
        export_df = pd.DataFrame(self.mem).transpose().reset_index()
        if tidy:
            export_df['steps'] = export_df['log'].map(lambda x: list(x.keys()))
            for v in self.logging_vars:
                export_df[v] = export_df['log'].map(
                    lambda x: [x[k].get(v) for k in x.keys()]
                )
            del export_df['log']
            export_df = export_df.explode(
                ['steps'] + list(self.logging_vars), ignore_index=True)
        return export_df
    
    def save(self, path):
        export = {}
        export['result_dir'] = self.result_dir
        export['log_dir'] = self.log_dir
        export['configs'] = self.configs
        export['mem'] = self.mem
        export['current_log'] = self.current_log
        export['logging_vars'] = list(self.logging_vars)
        with open(path, 'w') as f:
            json.dump(export, f)
            
    def delete_batch(self, batch_name, filter_field='experiment_name'):
        to_delete = []
        for k in self.mem.keys():
            if self.mem[k][filter_field] == batch_name:
                to_delete.append(k)
        for k in to_delete:
            del self.mem[k]
            
def load_logger(path):
    with open(path, 'r') as f:
        logger_import = json.load(f)
    log_date = logger_import['log_dir'].replace(logger_import['result_dir']+'/', '')
    logger = LightLogger(logger_import['result_dir'], log_date=log_date)
    logger.set_configs(logger_import['configs'])
    logger.mem = logger_import['mem']
    logger.current_log = logger_import['current_log']
    logger.logging_vars = set(logger_import['logging_vars'])
    print("Loading logger complete!")
    return logger
    