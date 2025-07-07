import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Set up CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
from datasets import load_dataset
from transformer_lens import HookedTransformer
import circuitsvis as cv

# Disable gradients for inference
torch.set_grad_enabled(False)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hallucination_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLaVAHallucinationAnalyzer:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_id = model_id
        self.results_dir = Path("analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Using device: {self.device}")
        self._load_models()
    
    def _load_models(self):
        """Load and initialize LLaVA models"""
        logger.info("Loading LLaVA models...")
        
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
        
        logger.info("Models loaded successfully")
    
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
        
        if len(image_token_indices) == 0:
            raise ValueError("No <image> token found in prompt")
        
        idx = image_token_indices[0].item()
        
        # Combine embeddings
        text_before = text_embeds[:, :idx, :]
        text_after = text_embeds[:, idx + 1:, :]
        inputs_embeds = torch.cat([text_before, image_features, text_after], dim=1)
        
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
        position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.long, device=self.device).unsqueeze(0)
        
        return inputs_embeds, attention_mask, None, position_ids, idx
    
    def analyze_single_image(self, image, question, image_name, save_plots=True):
        """Analyze a single image-question pair"""
        logger.info(f"Analyzing image: {image_name}, question: {question}")
        
        # Get model inputs
        inputs_embeds, attention_mask, labels, position_ids, image_idx = self.get_llm_input_embeddings(image, question)
        del self.llava
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
        
        # Analyze patterns
        metrics = self._compute_metrics(cache, image_idx, question, generated_text)
        
        if save_plots:
            self._create_visualizations(cache, image_idx, image_name, inputs_embeds, question, metrics)
            self._plot_generation_tokens(inputs_embeds)
        
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
            'cat_prob_final': float(cat_probs[-1]),
            'dog_prob_final': float(dog_probs[-1]),
            'max_dog_prob': float(np.max(dog_probs)),
            'max_cat_prob': float(np.max(cat_probs)),
            'prob_difference': float(dog_probs[-1] - cat_probs[-1]),
            'layer_predictions': layer_predictions.tolist()
        }
        
        return metrics

    def _generate_next_tokens(self, inputs_embeds, max_new_tokens, top_k):
        generation_analysis = []
        generated_token_ids = []
        
        current_inputs_embeds = inputs_embeds
        
        unembed = self.hooked_llm.W_U.to(dtype=torch.float32)
        
        for i in range(max_new_tokens):
            print(f"\n{'='*20} Analyzing for Generation Step {i+1} {'='*20}")
            
            logits, cache = self.hooked_llm.run_with_cache(
                current_inputs_embeds, 
                start_at_layer=0, 
                remove_batch_dim=True
            )
        
            resid_stack, labels = cache.accumulated_resid(layer=-1, incl_mid=True, return_labels=True)
            resid_stack_ln = cache.apply_ln_to_stack(resid_stack, layer=-1, pos_slice=-1)
            token_logits = resid_stack_ln @ unembed
            
            _, topk_indices = torch.topk(token_logits, k=top_k, dim=-1)
        
            step_top_tokens = [[
                [self.tokenizer.decode([idx]) for idx in topk_indices[layer, pos]]
                for pos in range(topk_indices.shape[1])
            ] for layer in range(topk_indices.shape[0])]
            
            generation_analysis.append(step_top_tokens)
        
            next_token_logits = logits[0, -1, :]
            
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            generated_token_ids.append(next_token_id.item())
            
            generated_token_str = self.tokenizer.decode([next_token_id])
            print(f"Current Sequence Length: {current_inputs_embeds.shape[1]}") # Use shape[1] for seq len
            print(f"==> Generated Token: '{generated_token_str}' (ID: {next_token_id.item()})")
            
            next_token_embed = self.hooked_llm.W_E[next_token_id]
            
            new_token_embed_3d = next_token_embed.unsqueeze(0).unsqueeze(0)
            
            current_inputs_embeds = torch.cat(
                [current_inputs_embeds, new_token_embed_3d],
                dim=1
            )
            torch.cuda.empty_cache()
            if next_token_id.item() == 2:
                break
        full_generated_text = self.tokenizer.decode(generated_token_ids)
        print(f"Full Generated Text: {full_generated_text}")
        return generation_analysis, generated_token_ids

    def _plot_generation_tokens(self, inputs_embeds, max_new_tokens = 10, top_k = 5):
        generation_analysis, generated_token_ids = self._generate_next_tokens(inputs_embeds, max_new_tokens, top_k)
        num_layers = len(generation_analysis[0])
        num_steps = len(generation_analysis)
        prompt_len = inputs_embeds.shape[1]
        
        token_grid = []
        match_grid = []
        
        for i, step_analysis in enumerate(generation_analysis):
            final_generated_token_str = self.tokenizer.decode([generated_token_ids[i]]).strip()
            current_seq_len = prompt_len + i
            final_pos_idx = current_seq_len - 1
        
            step_tokens = []
            match_row = []
            for layer_idx in range(num_layers):
                token_str = step_analysis[layer_idx][final_pos_idx][0].strip()
                step_tokens.append(token_str)
                match_row.append(token_str == final_generated_token_str)
            token_grid.append(step_tokens)
            match_grid.append(match_row)
        
        token_grid = np.array(token_grid).T
        match_grid = np.array(match_grid).T
        
        layer_indices = np.arange(0, num_layers, 2)
        token_grid = token_grid[layer_indices, :]
        match_grid = match_grid[layer_indices, :]
        
        colors = np.where(match_grid, '#FFD700', '#E0E0E0')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for y in range(token_grid.shape[0]):
            for x in range(token_grid.shape[1]):
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=colors[y, x]))
                text_color = 'black'
                ax.text(
                    x, y,
                    f"'{token_grid[y, x]}'",
                    ha='center',
                    va='center',
                    fontsize=8,
                    color=text_color,
                    fontweight='bold' if match_grid[y, x] else 'normal'
                )
        
        ax.set_xlabel("Generation Step", fontsize=12)
        ax.set_ylabel("Layer", fontsize=12)
        ax.set_title("Layer-wise Predicted Tokens (Every 2nd Layer)", fontsize=14, pad=10)
        
        ax.set_xticks(np.arange(num_steps))
        ax.set_yticks(np.arange(len(layer_indices)))
        ax.set_yticklabels([f"Layer {i}" for i in layer_indices])
        ax.invert_yaxis()
        ax.set_xlim(-0.5, num_steps - 0.5)
        ax.set_ylim(len(layer_indices) - 0.5, -0.5)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f"{image_name}_token_generation.png", bbox_inches='tight')
        plt.close()
    
    def _create_visualizations(self, cache, image_idx, image_name, inputs_embeds, question, metrics, layers_to_visualize = [1, 11, 21, 31]):
        """Create and save visualization plots"""
        # Token probability evolution
        fig, ax = plt.subplots(figsize=(12, 8))
        
        layer_predictions = np.array(metrics['layer_predictions'])
        target_tokens = ["cat", "dog"]
        
        for i, token in enumerate(target_tokens):
            ax.plot(range(len(layer_predictions)), layer_predictions[:, i], 
                   label=token, marker='o', linewidth=2)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Token Probability')
        ax.set_title(f'Token Probability Evolution - {image_name}\nQuestion: {question}')
        ax.legend()
        ax.grid(True, alpha=0.3)
                
        plt.tight_layout()
        plt.savefig(self.results_dir / f"{image_name}_token_evolution.png", bbox_inches='tight')
        plt.close()

        # Attention patterns
        for layer_to_visualize in layers_to_visualize:
            tokens_to_show = 10
            
            attention_pattern = cache["pattern", layer_to_visualize, "attn"]
            
            product = inputs_embeds @ self.language_model.model.embed_tokens.weight.T.to(self.device)  
            llama_str_tokens = self.hooked_llm.to_str_tokens(product.argmax(dim=-1)[0])
            
            attention_html = cv.attention.attention_patterns(tokens=llama_str_tokens[-tokens_to_show:], 
            										attention=attention_pattern[:, -tokens_to_show:, -tokens_to_show:])
                
            with open(self.results_dir / f"{image_name}_{layer_to_visualize}_attention.html", 'w') as f:
                f.write(attention_html._repr_html_())
    
    def run_full_analysis(self, num_images=15):
        """Run complete analysis on multiple images with different prompts"""        
        dataset = load_dataset("ULZIITOGTOKH/cat_images", split=f"train[:{num_images}]")
        
        choices = ["cat", "dog"]
        
        questions = [
            f"What colour is the {choices[0]}?",
            f"What colour is the {choices[1]}?",
        ]
        
        all_results = []
        
        for i, data in enumerate(dataset):
            image = data["image"]
            
            for j, question in enumerate(questions):
                result = self.analyze_single_image(
                    image, question, f"image_{i:02d}_{choices[j]}", save_plots=True
                )
                all_results.append(result)
                
                logger.info(f"Completed analysis for image {i}, question: {question}")
                            
        self._save_results(all_results)        
        return all_results
    
    def _save_results(self, results):
        """Save analysis results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"analysis_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    

if __name__ == "__main__":
    analyzer = LLaVAHallucinationAnalyzer()
    
    results = analyzer.run_full_analysis(num_images=1)
