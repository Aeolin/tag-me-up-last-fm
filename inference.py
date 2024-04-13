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

project = "tag-me-up-daddy"
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
ft_model = PeftModel.from_pretrained(base_model, "mistral-tag-me-up-daddy/checkpoint-500")


while True:
    song = input('>>> ').strip()
    if(song.startswith("https://genius.com")):
        html = requests.get(song)
        fixed_html = BeautifulSoup(html.content, 'html.parser')
        dom = etree.HTML(str(fixed_html))
        lyrics = dom.xpath("//div[@data-lyrics-container='true']")
        text = ''
        for section in lyrics:
            if section.text != None and section.text.strip() != '':
                text += section.text

            for child in section:
                if(child.tag == 'br'):
                    text += "\r\n" 
                elif child.tag == 'a':
                    text += child[0].text
                elif child.text != None and child.text.strip() != '':
                    text += child.text 

                if child.tail != None and child.tail.strip() != '':
                    text += child.tail
        song = text


    prompt = f"[INST]<<SYS>>Return the tag of the given Lyrics as JSON<</SYS>>{song}[/INST]"
    eval_prompt = f"<s>{prompt}"
    model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    ft_model.eval()
    with torch.no_grad():
        result = eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0], skip_special_tokens=True)
        result = result.replace(prompt, '')
        print("<<< " + result)