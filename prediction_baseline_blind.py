import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import transformers
from datetime import datetime
import requests
from tqdm import tqdm
import random
from multiprocessing import Pool, set_start_method
import os

def process_song(song):
    eval_prompt = format_prompt(song)
    model_input = eval_tokenizer(f"<s>{eval_prompt}", return_tensors="pt").to("cuda")
    
    with torch.no_grad(): 
        result = eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0], skip_special_tokens=True)
        result = result.replace(eval_prompt, '').strip()
        try:
            song['RawPrediction'] = result 
            tags = json.loads(result)
            tags = [tag.strip().lower() for tag in tags['tags']]
            song['PredictedTags'] = tags
        except Exception as error:
            song['error'] = str(error)  # Store the error in the song dictionary
    
    return song


if __name__ == "__main__":
    songs = None
    with open('evaluation_set.json', 'rt', encoding='utf8') as infile:
        songs = json.load(infile)

    set_start_method('spawn')
    with Pool(processes=4) as pool:  # Adjust number of processes based on your system's capabilities 
        results = list(tqdm(pool.imap(process_song, songs), total=len(songs)))
        with open('evaluation_results/baseline-blind/predictions.json', 'wt', encoding='utf8') as outfile:
            json.dump(results, outfile, indent=2)
else:
    os.environ['HF_TOKEN'] = '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'

    project = "tag-me-up-daddy-3"
    base_model_name = "mistral"
    run_name = base_model_name + "-" + project
    output_dir = "./" + run_name


    base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,  # Mistral, same as before
        quantization_config=bnb_config,  # Same quantization config as before
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
    ft_model = base_model#PeftModel.from_pretrained(base_model, "mistral-tag-me-up-daddy-3/checkpoint-500")

    def format_prompt(song):
        return f'[INST]<<SYS>>Tag the song based on the lyrics, only respond in json {{"tags": []}}<</SYS>>\n[{song["PrimaryArtistName"]}]\n{song["Lyrics"]}[/INST]'

    ft_model.eval()
