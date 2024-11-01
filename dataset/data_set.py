from sentencepiece import SentencePieceProcessor

from model import myGPTConfig
import numpy as np
import os
import sentencepiece as spm
import sys
import torch

config = myGPTConfig()
Learn_Text = './fanrenxiuxianchuan.txt'

def train_model(fname,prefix):
    spm.SentencePieceTrainer.Train(
        input=fname,model_prefix=prefix,vocab_size=config.vocab_size,)

# load tokenizer model
def load_tokenizer(my_model_file):
    sp = spm.SentencePieceProcessor()
    if not sp.Load(model_file=my_model_file):
        return False,None
    else:
        return True,sp



def split_file_for_train_and_target(text_file,split_ratio):
    with open(text_file, 'r') as file:
        data = file.read()
    
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx],data[split_idx:]
    

def gen_dataset(text_file,my_model_file):
    #sp:SentencePieceProcessor = spm.SentencePieceProcessor()
    #sp.Load(model_file=my_model_file)
    #if not sp.Load(model_file=my_model_file):
        #print(f"Failed:{my_model_file} load unsuccessfully")
        #sys.exit(1)
    flag,sp = load_tokenizer(my_model_file)
    if not flag:
        print(f"load tokenizer model from: {my_model_file} failed")
        sys.exit(1)

    split_ratio = config.train_data_split_ratio
    train_text, target_text = split_file_for_train_and_target(text_file,split_ratio)
    #print(sp.EncodeAsPieces("我在小学时就爱读课外书"))
    #print(sp.EncodeAsIds("为了使得读者易于分辨"))
    #print(sp.EncodeAsPieces("身上盖着的旧棉被"))
    #print(sp.EncodeAsIds("临走前韩父反复嘱咐韩立"))
    encode_and_save(sp,train_text,'train')
    encode_and_save(sp,target_text,'target')


def encode_and_save(sp:SentencePieceProcessor,content,prefix):
    token_ids = sp.Encode(content,out_type=int)
    #print(token_ids[9900:10000])
    #print(sp.DecodeIds((token_ids[9900:10000])))
    print(f"data splited of {prefix} has {len(token_ids)} tokens")
    token_ids = np.array(token_ids, dtype=np.int32)
    token_ids.tofile(os.path.join(os.path.dirname(__file__),f"{prefix}.dat"))

if __name__=='__main__':
    # generate two files
    # .model: contains the trained segmentation model
    # .vocab: contains the vocabulary of the model
    train_model(Learn_Text,"mydata")
    gen_dataset(Learn_Text, "mydata.model")

    #load_tokenizer("./mydata.model")

