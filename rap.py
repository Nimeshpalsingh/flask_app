from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = 'gpt2-medium'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_rap_lyrics(keywords, style='modern', length=16):
    # Create a structured and clear prompt
    prompt = (
        f"Write a {style} rap song that is humorous and uses the following keywords: "
        f"{', '.join(keywords)}. The rap should be {length} lines long and each line should rhyme. "
        "Make it clear that the lyrics are meant to entertain and are not offensive.\n"
        "Here are the lyrics:\n"
    )
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text
    outputs = model.generate(
        inputs,
        max_length=length * 15,  # Allow enough space for the rap
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    
    # Decode the generated text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated rap lyrics from the text
    generated_rap = text[len(prompt):].strip()
    
    # Ensure the generated rap has the desired number of lines
    rap_lines = generated_rap.split('\n')
    if len(rap_lines) > length:
        generated_rap = '\n'.join(rap_lines[:length])
    
    return generated_rap

keywords = ['struggle', 'grind', 'overcome']
lyrics = generate_rap_lyrics(keywords, length=16)
print(lyrics)