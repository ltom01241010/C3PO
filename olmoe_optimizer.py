import torch
import numpy as np
import json 
import copy
from transformers import OlmoeForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from fvcore.nn import FlopCountAnalysis, flop_count_table
from collections import Counter
from tqdm import tqdm

# =========================
# Monkey-Patch OlmoeSparseMoeBlock.forward method to support custom routing weights
# =========================

try:
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
except ImportError:
    raise ImportError("Could not import OlmoeSparseMoeBlock, please check your transformers version.")

def custom_moe_forward(self, hidden_states: torch.Tensor):
    """
    Custom forward method for MoE module.
    After computing routing weights, if the module attribute self.custom_routing_weights is set,
    it replaces the routing weights of the last token of each sample with the custom probability distribution.
    
    Args:
        hidden_states: Input tensor of shape [batch_size, sequence_length, hidden_dim]
        
    Returns:
        Tensor of shape [batch_size, sequence_length, hidden_dim] and router logits
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_flat = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states_flat)
    routing_weights = torch.nn.functional.softmax(router_logits, dim=1)
    
    # If custom routing weights are set, override the routing weights for the last token of each sample
    if hasattr(self, 'custom_routing_weights') and self.custom_routing_weights is not None:
        custom_weights_list = [self.custom_routing_weights.get(f"expert_{i}", 1e-10)
                               for i in range(self.num_experts)]
        custom_weights_tensor = torch.tensor(custom_weights_list, dtype=routing_weights.dtype, device=routing_weights.device)
        custom_weights_tensor = custom_weights_tensor / custom_weights_tensor.sum()
        indices = [i * sequence_length + (sequence_length - 1) for i in range(batch_size)]
        routing_weights[indices] = custom_weights_tensor.unsqueeze(0).expand(batch_size, -1)
    
    routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=1)
    if self.norm_topk_prob:
        routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=1, keepdim=True)
    routing_weights_topk = routing_weights_topk.to(hidden_states_flat.dtype)

    final_hidden_states = torch.zeros_like(hidden_states_flat)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, token_indices = torch.where(expert_mask[expert_idx])
        if token_indices.numel() == 0:
            continue
        current_state = hidden_states_flat[token_indices]
        current_hidden = expert_layer(current_state) * routing_weights_topk[token_indices, idx].unsqueeze(-1)
        final_hidden_states.index_add_(0, token_indices, current_hidden)
    final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits

# Apply the monkey patch
OlmoeSparseMoeBlock.forward = custom_moe_forward

# =========================
# Helper Functions
# =========================

def extract_answer_option(text):
    """
    Extract the option letter (e.g., "A", "B", "C", "D") from the text.
    First tries to find an option after "Answer:", if not found, looks for the first occurrence
    of an option letter in the text. Returns an empty string if extraction fails.
    
    Args:
        text: The text to extract the answer option from
        
    Returns:
        A string containing the option letter, or an empty string if not found
    """
    idx = text.find("Answer:")
    if idx != -1:
        substring = text[idx+len("Answer:"):].strip()
        for token in substring.split():
            token = token.strip().upper()
            if token and token[0] in "ABCD":
                return token[0]
    for token in text.split():
        token = token.strip().upper()
        if token in ["A", "B", "C", "D"]:
            return token
        elif token.startswith(("A.", "B.", "C.", "D.")) and len(token) <= 3:
            return token[0]
        elif token.startswith(("A)", "B)", "C)", "D)")) and len(token) <= 3:
            return token[0]
    import re
    pattern = r'\b([A-D])[\.|\)]?\b'
    matches = re.findall(pattern, text.upper())
    if matches:
        return matches[0]
    return ""

def extract_routing_info(text, model, tokenizer, batch_size=512, max_length=64):
    """
    Extract routing information for each token based on the input text.
    Only extracts routing information for the last token and stores it in the last_token_routing field,
    while also generating output text (max_length set to 64).
    
    Args:
        text: The input text
        model: The OLMoE model
        tokenizer: The tokenizer for the model
        batch_size: Maximum batch size for processing tokens
        max_length: Maximum length for generating text
        
    Returns:
        A dictionary containing routing information and generated text
    """
    tokens = tokenizer(text, truncation=False)["input_ids"]
    expert_counter = Counter()
    last_token_routing = None

    with torch.no_grad():
        for i in tqdm(range(0, len(tokens), batch_size), desc="Processing tokens"):
            batch_tokens = tokens[i:min(i + batch_size, len(tokens))]
            input_ids = torch.tensor(batch_tokens).reshape(1, -1).to(model.device)
            outputs = model(input_ids=input_ids, output_router_logits=True)
            all_layers_logits = outputs["router_logits"]
            all_layers_probs = [torch.nn.functional.softmax(layer_logits.float(), dim=-1).cpu().numpy() 
                                for layer_logits in all_layers_logits]
            for layer_probs in all_layers_probs:
                top_experts = np.argsort(-layer_probs, axis=-1)
                expert_counter.update(top_experts.flatten().tolist())

        last_token_idx = len(tokens) - 1
        layers_info = []
        n_layers = len(all_layers_probs)
        for layer_idx in range(n_layers):
            token_position = last_token_idx % batch_size
            distribution = all_layers_probs[layer_idx][token_position]
            sorted_experts = sorted(enumerate(distribution), key=lambda x: x[1], reverse=True)
            layer_info = {
                "layer": layer_idx,
                "routing_weights": {f"expert_{expert_id}": float(weight)
                                      for expert_id, weight in sorted_experts}
            }
            layers_info.append(layer_info)
        last_token_routing = {
            "token_id": tokens[last_token_idx],
            "token_text": tokenizer.decode([tokens[last_token_idx]]),
            "layers": layers_info
        }

    input_ids = torch.tensor(tokens).reshape(1, -1).to(model.device)
    generated_output = model.generate(input_ids=input_ids, max_length=max_length)
    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

    return {
        'last_token_routing': last_token_routing,
        'expert_counts': {int(k): int(v) for k, v in expert_counter.items()},
        'tokens': tokens,
        'tokenizer': tokenizer.name_or_path,
        'generated_text': generated_text,
        'input_text': text
    }

def save_results_to_json(results, filename="all_results.json"):
    """
    Save results to a JSON file.
    
    Args:
        results: The results to save
        filename: The filename to save to
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {filename}")

def print_analysis_results(results):
    """
    Print analysis results in a human-readable format.
    
    Args:
        results: The results to print
    """
    total_selections = sum(results['expert_counts'].values())
    print("\nExpert utilization rates:")
    for expert_id, count in sorted(results['expert_counts'].items()):
        percentage = count / total_selections * 100
        print(f"Expert {expert_id}: {percentage:.2f}%")
    print("\nLast token routing weights for each layer:")
    last_routing = results.get("last_token_routing", {})
    if last_routing:
        print(f"\nToken ID: {last_routing['token_id']}, Text: '{last_routing['token_text']}'")
        for layer_info in last_routing.get("layers", []):
            print(f"  Layer {layer_info['layer']}:")
            for expert_id, weight in layer_info['routing_weights'].items():
                print(f"    {expert_id}: {weight:.4f}")
    print("\nGenerated Text:")
    print(results['generated_text'])

def load_multiple_reference_files(file_paths):
    """
    Load multiple reference dataset files and merge them into a single reference set.
    
    Args:
        file_paths: List of file paths to load
        
    Returns:
        A dictionary containing the combined reference cases
    """
    combined_references = {}
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
                print(f"Loaded {len(dataset)} reference cases from {file_path}")
                prefix = file_path.split("_")[1].split(".")[0]
                prefixed_dataset = {f"{prefix}_{key}": value for key, value in dataset.items()}
                combined_references.update(prefixed_dataset)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
    print(f"Loaded a total of {len(combined_references)} reference cases")
    return combined_references

def detect_custom_routing_usage(model):
    """
    Check if custom routing weights are being used in any decoder layer of the model.
    
    Args:
        model: The OLMoE model
        
    Returns:
        True if custom routing weights are being used, False otherwise
    """
    used = False
    for layer in model.model.layers:
        if hasattr(layer.mlp, 'custom_routing_weights') and layer.mlp.custom_routing_weights is not None:
            used = True
            break
    if used:
        print("Detected custom routing weights in use.")
    else:
        print("No custom routing weights detected.")
    return used

def re_infer_case(case, reference_cases, embedder, model, tokenizer, max_length=64):
    """
    Re-infer a case by updating routing weights (optimizing only the last five layers),
    generating new text, and outputting information about the question, original inference answer,
    correct answer, neighboring cases, and optimized inference results.
    
    Args:
        case: The case to re-infer
        reference_cases: Dictionary of reference cases
        embedder: The sentence embedder to use for finding similar cases
        model: The OLMoE model
        tokenizer: The tokenizer for the model
        max_length: Maximum length for generating text
        
    Returns:
        A dictionary containing the re-inference results
    """
    result = {}
    question = case.get("input_text", "")
    result["question"] = question
    result["correct_answer"] = case.get("correct_answer", "N/A").strip().upper()
    result["model_answer"] = case.get("model_answer", "N/A").strip().upper()
    result["is_correct"] = case.get("is_correct", None)
    result["original_inference"] = result["model_answer"]
    result["original_output_text"] = case.get("generated_text", "")

    # Get embedding for the current question
    case_embedding = embedder.encode(question, convert_to_tensor=True)

    # Find neighboring cases
    ref_questions = []
    ref_keys = []
    for key, one_case in reference_cases.items():
        q_text = one_case.get("input_text", "")
        ref_questions.append(q_text)
        ref_keys.append(key)
    ref_embeddings = embedder.encode(ref_questions, convert_to_tensor=True)
    cosine_scores = util.cos_sim(case_embedding, ref_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=3)
    top_indices = top_results[1].tolist()
    top_scores = top_results[0].tolist()

    neighbors = []
    print(f"\nProcessing case question:\n  {question}")
    for idx, score in zip(top_indices, top_scores):
        similar_key = ref_keys[idx]
        similar_question = reference_cases[similar_key].get("input_text", "")
        neighbor_correct = reference_cases[similar_key].get("correct_answer", "N/A").strip().upper()
        neighbors.append({
            "case_id": similar_key,
            "question": similar_question,
            "correct_answer": neighbor_correct,
            "similarity": float(score)
        })
        print(f"  Neighbor case {similar_key}: Similarity {score:.4f}, Question: {similar_question}  Correct answer: {neighbor_correct}")
    result["neighbors"] = neighbors

    # -----------------------------
    # Modified part: Optimize only the last five layers' routing weights
    # -----------------------------
    routing_info = case.get("last_token_routing", None)
    if routing_info is None:
        print("Current case has no last_token_routing, skipping.")
        return None
    layers = routing_info.get("layers", [])
    if len(layers) < 1:
        print("Current case has fewer than 1 layer, skipping.")
        return None

    # If fewer than 5 layers, optimize all; otherwise, take the last 5
    if len(layers) < 5:
        target_layers = layers
    else:
        target_layers = layers[-5:]

    # Save original target layers' routing weights
    result["original_routing_target_layers"] = copy.deepcopy(target_layers)

    updated_layers = []
    for layer in target_layers:
        updated = copy.deepcopy(layer)
        weighted_sum = None
        total_weight = 0.0
        for sim_score, idx in zip(top_scores, top_indices):
            neighbor_key = ref_keys[idx]
            neighbor_case = reference_cases[neighbor_key]
            neighbor_routing = neighbor_case.get("last_token_routing", None)
            if neighbor_routing is None:
                continue
            neighbor_layers = neighbor_routing.get("layers", [])
            neighbor_layer_info = next((l for l in neighbor_layers if l["layer"] == layer["layer"]), None)
            if neighbor_layer_info is None:
                continue
            weights = np.array([neighbor_layer_info["routing_weights"].get(f"expert_{i}", 0.0)
                                for i in range(64)], dtype=np.float32)
            if weighted_sum is None:
                weighted_sum = sim_score * weights
            else:
                weighted_sum += sim_score * weights
            total_weight += sim_score
        if weighted_sum is not None and total_weight > 0:
            new_weights = weighted_sum / total_weight
            new_weights = new_weights / new_weights.sum()
            new_weights_dict = {f"expert_{i}": float(new_weights[i]) for i in range(64)}
            updated["routing_weights"] = dict(sorted(new_weights_dict.items(), key=lambda item: item[1], reverse=True))
            print(f"Updated routing weights for layer {layer['layer']} of the current case.")
        else:
            print(f"Unable to update weights for layer {layer['layer']} of the current case (missing corresponding data).")
        updated_layers.append(updated)
    result["updated_routing_target_layers"] = updated_layers

    tokens = case.get("tokens", None)
    if tokens is None:
        input_text = case.get("input_text", "")
        tokens = tokenizer.encode(input_text, truncation=False)
        print("Generated tokens from input_text.")
    result["tokens"] = tokens

    english_prompt = ("Answer with only a single letter (A, B, C, or D) representing the correct option. "
                      "Do not explain your reasoning. Just output the letter of the answer. ")
    modified_input_text = english_prompt + question
    input_ids = tokenizer(modified_input_text, return_tensors="pt").input_ids.to(model.device)

    # Set the updated routing weights to the corresponding decoder layers of the model
    for updated_layer in updated_layers:
        layer_idx = updated_layer["layer"]
        model.model.layers[layer_idx].mlp.custom_routing_weights = updated_layer["routing_weights"]

    try:
        new_generated_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
        )
        new_generated_text = tokenizer.decode(new_generated_output[0], skip_special_tokens=True)
        print("\nFull text generated using updated routing weights:")
        print(new_generated_text)
    except Exception as e:
        print("\nError during generation, performing standard generation. Error message:", e)
        new_generated_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
        )
        new_generated_text = tokenizer.decode(new_generated_output[0], skip_special_tokens=True)
        print("\nFull text generated using standard method:")
        print(new_generated_text)
    
    routing_used = detect_custom_routing_usage(model)
    result["custom_routing_used"] = routing_used

    result["new_generated_text_full"] = new_generated_text
    optimized_inference = extract_answer_option(new_generated_text)
    result["optimized_inference"] = optimized_inference

    result["original_inference"] = case.get("model_answer", "").strip().upper()
    correct_ans = case.get("correct_answer", "").strip().upper()
    result["correct_inference"] = correct_ans

    result["optimized_is_correct"] = (optimized_inference == correct_ans)

    # Reset the target layers' custom_routing_weights to avoid affecting subsequent generations
    for updated_layer in updated_layers:
        layer_idx = updated_layer["layer"]
        model.model.layers[layer_idx].mlp.custom_routing_weights = None

    return result

# =========================
# Main Program
# =========================

def main():
    """
    Main function to run the demo.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "allenai/OLMoE-1B-7B-0125-Instruct"
    print(f"Loading model {model_name}...")
    model = OlmoeForCausalLM.from_pretrained(
        model_name,
        device_map={'': 0},
        torch_dtype=torch.float16
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded successfully!")

    # Demo 1: Generate routing information for a sample text
    print("\n===== Demo 1: Routing Information Analysis =====")
    sample_text = "This is a test to analyze routing information in the OLMoE model."
    print(f"Analyzing sample text: '{sample_text}'")
    results = extract_routing_info(sample_text, model, tokenizer, max_length=64)
    save_results_to_json(results, filename="routing_results.json")
    print_analysis_results(results)

    # Demo 2: Re-inference optimization
    print("\n===== Demo 2: Routing Optimization for Multiple-Choice Questions =====")
    
    # Load the evaluation cases
    try:
        with open("arc_challenge_routing_results.json", "r", encoding="utf-8") as f:
            evaluation_cases = json.load(f)
        print(f"Loaded {len(evaluation_cases)} evaluation cases.")
    except FileNotFoundError:
        print("Evaluation cases file not found. Skipping Demo 2.")
        return
    
    # Load the reference cases for routing optimization
    reference_files = [
        "instruct_openbookqa_correct_routing_results.json",
        "instruct_sciq_correct_routing_results.json",
    ]
    
    # Check if reference files exist
    missing_files = [f for f in reference_files if not os.path.exists(f)]
    if missing_files:
        print(f"Missing reference files: {missing_files}. Skipping Demo 2.")
        return
    
    reference_cases = load_multiple_reference_files(reference_files)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    correct_evaluation = []
    incorrect_evaluation = []
    total = 0
    max_cases = 10  # Demo with just 10 cases for brevity
    
    print(f"\nProcessing {max_cases} cases for demonstration...")
    for idx, (case_id, one_case) in enumerate(evaluation_cases.items()):
        if idx >= max_cases:
            break
        print(f"\n=========== Processing case {case_id} ===========")
        eval_result = re_infer_case(one_case, reference_cases, embedder, model, tokenizer, max_length=200)
        if eval_result is None:
            continue
        total += 1
        if one_case.get("is_correct", False):
            correct_evaluation.append({"case_id": case_id, **eval_result})
        else:
            incorrect_evaluation.append({"case_id": case_id, **eval_result})

    print(f"\nTotal evaluated cases: {total}")
    print(f"Correct cases: {len(correct_evaluation)}; Incorrect cases: {len(incorrect_evaluation)}")

    originally_correct = 0
    originally_incorrect = 0
    improved = 0       
    worsened = 0       
    still_correct = 0  
    still_incorrect = 0  

    all_evaluation = correct_evaluation + incorrect_evaluation
    for case in all_evaluation:
        original_answer = case["original_inference"]
        optimized_answer = case["optimized_inference"]
        correct_answer = case["correct_inference"]
        
        original_is_correct = (original_answer == correct_answer)
        if original_is_correct:
            originally_correct += 1
        else:
            originally_incorrect += 1
            
        optimized_is_correct = (optimized_answer == correct_answer)
        
        if original_is_correct and optimized_is_correct:
            still_correct += 1
        elif original_is_correct and not optimized_is_correct:
            worsened += 1
        elif not original_is_correct and optimized_is_correct:
            improved += 1
        else:
            still_incorrect += 1

    print("\n========== Optimization Results ==========")
    print(f"Total cases: {total}")
    print(f"Originally correct cases: {originally_correct} ({originally_correct/total*100:.2f}%)")
    print(f"Originally incorrect cases: {originally_incorrect} ({originally_incorrect/total*100:.2f}%)")
    print(f"Optimized correct cases: {still_correct + improved} ({(still_correct + improved)/total*100:.2f}%)")
    print(f"Optimized incorrect cases: {still_incorrect + worsened} ({(still_incorrect + worsened)/total*100:.2f}%)")
    print("\nDetailed optimization results:")
    print(f"Improved cases (wrong→right): {improved} ({improved/total*100:.2f}%)")
    print(f"Worsened cases (right→wrong): {worsened} ({worsened/total*100:.2f}%)")
    print(f"Maintained correct cases: {still_correct} ({still_correct/total*100:.2f}%)")
    print(f"Maintained incorrect cases: {still_incorrect} ({still_incorrect/total*100:.2f}%)")

    net_improvement = improved - worsened
    print(f"\nNet improvement: {'+' if net_improvement >= 0 else ''}{net_improvement} cases ({net_improvement/total*100:.2f}%)")

    optimization_stats = {
        "total_cases": total,
        "originally_correct": originally_correct,
        "originally_incorrect": originally_incorrect,
        "optimized_correct": still_correct + improved,
        "optimized_incorrect": still_incorrect + worsened,
        "improved_cases": improved,
        "worsened_cases": worsened,
        "still_correct_cases": still_correct,
        "still_incorrect_cases": still_incorrect,
        "net_improvement": net_improvement,
        "original_accuracy": originally_correct/total,
        "optimized_accuracy": (still_correct + improved)/total
    }

    save_results_to_json(optimization_stats, filename="optimization_statistics.json")
    print("\nDemo completed successfully! Results saved to optimization_statistics.json")


if __name__ == "__main__":
    import os
    
    print("OLMoE Routing Weight Optimizer Demo")
    print("===================================")
    print("This demo shows how to optimize the routing weights of OLMoE models")
    main()
