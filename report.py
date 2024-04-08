from transformers import OPTForCausalLM, OPTTokenizer
import torch
from datasets import load_dataset
from tqdm import tqdm
from rouge import Rouge
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "facebook/opt-1.3b"
model = OPTForCausalLM.from_pretrained(model_id).to(device)
tokenizer = OPTTokenizer.from_pretrained(model_id)

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt", padding=True)

max_length = model.config.max_position_embeddings
stride = 512

seq_len = encodings.input_ids.size(1)
nlls = []
ttfts = []
itls = []
throughputs = []
rouge_scorer = Rouge()
prev_end_loc = 0

for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc

    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_ids, labels=target_ids)
        ttft = time.time() - start_time

        generated_text = tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
        ground_truth = tokenizer.batch_decode(target_ids[:, trg_len:], skip_special_tokens=True)

        rouge_score = rouge_scorer.score(" ".join(generated_text), " ".join(ground_truth))

        neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
        ttfts.append(ttft)

        itl = ttft / (trg_len - 1)
        itls.append(itl)

        throughput = (trg_len - 1) / ttft
        throughputs.append(throughput)

    prev_end_loc = end_loc

    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())

print(f"Perplexity: {ppl.item()}")
print(f"Average TTFT: {sum(ttfts) / len(ttfts):.4f} seconds")
print(f"Average ITL: {sum(itls) / len(itls):.4f} seconds")
print(f"Average Throughput: {sum(throughputs) / len(throughputs):.2f} tokens/second")
print(f"ROUGE-L Score: {rouge_score['rouge-l']['f']:.4f}")
