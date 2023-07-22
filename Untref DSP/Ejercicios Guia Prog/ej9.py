import numpy as np
import ej8 as UT

class AudioStats():
    def __init__(self, audio_array):
        if len(audio_array.shape) == 2:
            self.audio_array = np.mean(audio_array, axis=tuple(range(audio_array.ndim - 1)))
        else:
            self.audio_array = audio_array
        
    def mean(self):
        return UT.mean(self.audio_array)
    
    def std_desv(self):
        return UT.des_est(self.audio_array)
    
    def rms(self):
        return UT.rms_value(self.audio_array)
    
    def get_stats(self):
        stats = {
            "mean" : self.mean(self.audio_array),
            "std_desv" : self.std_desv(self.audio_array),
            "rms" : self.rms(self.audio_array)
        }
        
        return stats
    
    