import os
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
from datasets import load_dataset
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

class LLaVA:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_id = model_id
        self.results_dir = Path("analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self._load_models()
    
    def _load_models(self):
        """Load and initialize LLaVA models"""
        
        # Load LLaVA
        self.llava = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        
        # Disable gradients
        for param in self.llava.parameters():
            param.requires_grad = False
        
        # Load processor
        self.processor = LlavaProcessor.from_pretrained(self.model_id, use_fast=True)
        self.tokenizer = self.processor.tokenizer
        
        # Get model components
        self.language_model = self.llava.language_model.eval()
        self.vision_tower = self.llava.vision_tower.to(self.device).eval()
        self.projector = self.llava.multi_modal_projector.to(self.device).eval()
        
        # Create HookedTransformer
        self.hooked_llm = HookedTransformer.from_pretrained(
            "llama-7b-hf",
            center_unembed=False,
            fold_ln=False,
            fold_value_biases=False,
            device=self.device,
            hf_model=self.language_model,
            tokenizer=self.tokenizer,
            center_writing_weights=False,
            dtype=torch.float16,
            vocab_size=self.language_model.config.vocab_size
        )
        
        for param in self.hooked_llm.parameters():
            param.requires_grad = False
            
    def get_llm_input_embeddings(self, image: Image.Image, text: str):
        """Extract features from image and create input embeddings"""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image", "image": image},
                ],
            },
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process image
        pixel_values = self.processor.image_processor(image, return_tensors='pt')['pixel_values'].to(self.device, torch.float16)
        clip_output = self.llava.vision_tower(pixel_values)
        image_features = self.llava.multi_modal_projector(clip_output.last_hidden_state)
        
        # Process text
        input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)
        embed_tokens = self.llava.language_model.model.embed_tokens.to(self.device)
        text_embeds = embed_tokens(input_ids)
        
        # Find image token position
        image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        image_token_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[1]
        
        idx = image_token_indices[0].item()
        
        # Combine embeddings
        text_before = text_embeds[:, :idx, :]
        text_after = text_embeds[:, idx + 1:, :]
        inputs_embeds = torch.cat([text_before, image_features, text_after], dim=1)
        
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
        position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.long, device=self.device).unsqueeze(0)
        
        return inputs_embeds, attention_mask, None, position_ids, idx
    
    def analyze_single_image(self, image, question, image_name):
        """Analyze a single image-question pair"""
        # Get model inputs
        inputs_embeds, attention_mask, labels, position_ids, image_idx = self.get_llm_input_embeddings(image, question)
        # Run model with cache
        logits, cache = self.hooked_llm.run_with_cache(inputs_embeds, start_at_layer=0, remove_batch_dim=True)
        
        # Generate response
        outputs = self.hooked_llm.generate(
            inputs_embeds,
            max_new_tokens=50,
            do_sample=True,
            return_type='tokens'
        )
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        metrics = self._compute_metrics(cache, image_idx, question, generated_text)
                
        return {
            'image_name': image_name,
            'question': question,
            'generated_text': generated_text,
            'metrics': metrics
        }
    
    def _compute_metrics(self, cache, image_idx, question, generated_text):
        """Compute metrics"""
        # Get residual stream
        resid_stack, labels = cache.accumulated_resid(layer=-1, incl_mid=True, return_labels=True)
        
        # Target tokens for analysis
        target_tokens = ["cat", "dog"]
        token_ids = [self.tokenizer.encode(token, add_special_tokens=False)[0] for token in target_tokens]
        
        # Compute token probabilities across layers
        unembed = self.hooked_llm.W_U
        layer_predictions = []
        
        for layer in range(len(resid_stack)):
            normalized = cache.apply_ln_to_stack(resid_stack[layer:layer+1], layer=-1, pos_slice=-1)
            logits = normalized @ unembed.to(dtype=torch.float32)
            probs = torch.softmax(logits, dim=-1)
            
            layer_token_probs = []
            for token_id in token_ids:
                prob = probs[0, -1, token_id].item()
                layer_token_probs.append(prob)
            
            layer_predictions.append(layer_token_probs)
        
        layer_predictions = np.array(layer_predictions)
        
        cat_probs = layer_predictions[:, 0]
        dog_probs = layer_predictions[:, 1]
        
        metrics = {
            'prob_difference': float(dog_probs[-1] - cat_probs[-1]),
            'layer_predictions': layer_predictions.tolist()
        }
        
        return metrics
    
    def load_data(self, num_images=15, dataset_name="ULZIITOGTOKH/cat_images"):
        dataset = load_dataset(dataset_name, split=f"train[:{num_images}]")
        choices = ["cat", "dog"]

        questions = [
            f"What colour is the {choices[0]}?",
            f"What colour is the {choices[1]}?",
        ]
        self.dataset = []
        for i, data in enumerate(dataset):            
            for j, question in enumerate(questions):
                self.dataset.append((data["image"], question, f"image_{i:02d}_{choices[j]}"))

    
    def run_full_analysis(self, num_images=15):
        """Run on multiple images with different prompts"""        
        self.load_data(num_images)
        
        all_results = []
        
        for image, question, image_path in self.dataset:
            result = self.analyze_single_image(
                image, question, image_path
            )
            all_results.append(result)
                            
        self._save_results(all_results)        
        return all_results
    
    def _save_results(self, results):
        """Save analysis results to JSON file"""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        results_file = self.results_dir / "analysis_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            

if __name__ == "__main__":
    llava = LLaVA()
    
    results = llava.run_full_analysis(num_images=200)
