'''
This is a hacky version of my code to extract all representations for various corpora
'''


import numpy as np
import os
import sys
import librosa as lib
import fairseq
import soundfile as sf
import torch
import torch.nn.functional as F
import glob
import os
import fire
from tqdm import tqdm

    
class FeatureExtractor():
    def __init__(
        self,
        device, # device is also passsed in
        encoder, # the encoder should be passed in
        WSJ_FLAG, 
        task_cfg=None,
        offset=False,
    ):
        # to make these visible to the rest of the class
        self.device=device
        self.encoder=encoder
        self.contextualized_features = {}
        self.mean_pooling=True
        self.rep_dct = {}
        self.label_lst = []
        self.WSJ_FLAG=WSJ_FLAG

    def load_time_stamps(self, filename, data_path): # this is a time stamp list for individual files
        time_stamp_lst=[]
        with open(os.path.join(data_path, filename), 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                if WSJ_FLAG == True:
                    time_stamp_lst.extend([line.split()[1:3] + line.split()[4:5]]) # doing thi because wsj has extra probability column
                else:
                    time_stamp_lst.append(line.split()[1:4])

        return time_stamp_lst 

    def extract_contextualized_embeddings(self, data_path, out_dir, out_name):
        '''
        this function will extract the embeddings
        it assumes that the filename has an associated time stamp list 
        '''
        
        ext = '*.%s' % self.file_type
        print('extension: '); print(ext)

        pbar = tqdm(glob.glob(os.path.join(data_path, '*.flac')))

        for filename in glob.glob(os.path.join(data_path, ext)): # must be flac foor librispeech,  mp3 for common voice
            basefile = os.path.splitext(filename)[0]
            print(basefile)
            curr_audio=lib.load(basefile + ext, sr=16000) # automatically downsample to 16000
            print('current audio file')
            in_data = torch.from_numpy(np.expand_dims(curr_audio[0], 0).astype("float32")).to(self.device)
            with torch.no_grad():
                in_rep = self.encoder.feature_extractor(in_data)
                n_frames = len(in_rep.transpose(1, 2).squeeze(0))
                encoder_out = encoder(in_data, features_only=True, mask=False) # getting the embedding itself       
                for layer_num, layer_rep in enumerate(encoder_out["layer_results"]):
                    self.contextualized_features[layer_num] = (
                        layer_rep[0].squeeze(1).cpu().numpy()  
                            )
                            
            print('current transcription file')
            print(basefile, ".txt")

            assert os.path.exists(basefile + "txt") == True # make sure that a each of the audio files actually has a transcriptiono associated with it

            '''
            the following should actually still be split up into a separate function
            '''
            time_stamp_lst=self.load_time_stamps(basefile + ".txt")
            num_layers = len(self.contextualized_features)
            for layer_num in range(num_layers):
                c_rep = self.contextualized_features[layer_num]
                for start_time, end_time, token in time_stamp_lst:
                    indices = self.get_segment_idx(start_time, end_time, len(c_rep))
                    if len(indices) > 1:
                        self.update_dct(indices, c_rep, rep_dct, layer_num)
                        if layer_num == 0 and label_lst is not None:
                            self.label_lst.append(token)
        print("length of label list: ", len(self.label_lst))
        np.save(os.path.join(out_dir, out_name + "_reps.npy"), self.rep_dct)
        np.save(os.path.join(out_dir, out_name + "_labels.npy"), self.label_lst)
 
    def update_dct(self, indices, rep_array, key):
        _ = self.rep_dct.setdefault(key, [])
        #print(len(rep_array))
        rep_array_masked = rep_array[indices] # the mean pooling might be what's causing an issue here
        if len(rep_array_masked) == 0:
            print('something went wrong, rep array len is 0')
        if self.mean_pooling:
            rep_array_masked = np.expand_dims(np.mean(rep_array_masked, 0), 0)
        self.rep_dct[key].append(rep_array_masked)


    def get_segment_idx(self, start_time, end_time, len_utt):
        stride_sec = 20/1000 
        start_id = int(np.floor(float(start_time) / stride_sec))
        end_id = int(np.ceil(float(end_time) / stride_sec))
        offset = 0
        start_id += offset
        end_id -= offset
        if end_id == start_id:
            end_id += 1
        if end_id == len_utt + 1:
            end_id = len_utt
          
        # I guess kind of a hacky fix for the end of the utterances for now?
        if start_id > len_utt:
            start_id = len_utt-1
        if end_id > len_utt + 1:
            #print('end_id longer than utterance')
            end_id = len_utt   
        assert end_id >= start_id

        return np.arange(start_id, end_id)

def save_rep(data_path, out_dir, out_name, cp):

    # data_path = "/fs/clip-scratch/jrodri20/librispeech/LibriSpeech/train-clean-100-flattened/files/"
    # out_dir = "/fs/clip-scratch/jrodri20/wav2vec/embeddings/librispeech/"
    # outname = librispeech_sample

    sys.stdout.write("torch availabliy: %s \n" % torch.cuda.is_available())    
    encoder, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp])
    encoder = encoder[0] # the extra [0] is just because it's wrapped in a list for some reason
    task_cfg = task.cfg # not 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    encoder.to(device) # moving model to gpu 

    extractor = FeatureExtractor(device, encoder)
    extractor.extract_contextualized_embeddings(data_path = data_path, out_dir=out_dir, out_name=out_name)
    
if __name__ == "__main__":
    fire.Fire()
