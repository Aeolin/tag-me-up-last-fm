import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import transformers
from datetime import datetime
from bs4 import BeautifulSoup
import requests
from lxml import etree
from tqdm import tqdm
import random

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
)

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
ft_model = PeftModel.from_pretrained(base_model, "mistral-tag-me-up-daddy-3/checkpoint-500")

def format_prompt(song):
    return f'[INST]<<SYS>>Tag the song based on the lyrics, only respond in json {{"tags": []}}<</SYS>>\n[{song['PrimaryArtistName']}]\n{song['Lyrics']}[/INST]'

songs = None
with open('evaluation_data/all_songs.json', 'rt', encoding='utf8') as infile:
    songs = json.load(infile)

def get_random():
    song = random.choice(songs)
    songs.remove(song)
    return song

random_selection = [get_random() for x in range(8000)]

for song in tqdm(random_selection, total=len(random_selection)):
    eval_prompt = format_prompt(song)
    model_input = eval_tokenizer(f"<s>{eval_prompt}", return_tensors="pt").to("cuda")

    ft_model.eval()
    with torch.no_grad(): 
        result = eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0], skip_special_tokens=True)
        result = result.replace(eval_prompt, '').strip()
        song['RawPrediction'] = result
        try:
            tags = json.loads(result)
            song['PredictedTags'] = tags['tags']
        except Exception as error:
            print(error)

with open('predictions_evaluation.json', 'wt', encoding='utf8') as outfile:
    json.dump(songs, outfile, indent=2)
