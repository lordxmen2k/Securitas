import os
import time
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from flask import Flask, render_template, request

# ART and Fairlearn imports
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate

# Set fixed seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

app = Flask(__name__)

############################################
# Define available models
############################################

MODEL_LOADERS = {
    "resnet50": models.resnet50,
    "vgg16": models.vgg16,
    "mobilenet_v2": models.mobilenet_v2,
    "densenet121": models.densenet121,
}

def download_and_save_model(model_name, model_path=None):
    """
    Download a pre-trained model by name, set it to evaluation mode, and save its state.
    """
    loader = MODEL_LOADERS.get(model_name.lower())
    if loader is None:
        raise ValueError(f"Model '{model_name}' not found. Choose from {list(MODEL_LOADERS.keys())}.")
    
    model = loader(pretrained=True)
    model.eval()  # Set to evaluation mode

    if model_path is None:
        # Get the absolute path to the "models" directory
        models_dir = os.path.join(os.path.dirname(__file__), "models")

        # Construct the full model path
        model_path = os.path.join(models_dir, f"{model_name.lower()}.pth")

    torch.save(model.state_dict(), model_path)
    print(f"{model_name} model saved as {model_path}")
    return model

############################################
# Functions for testing the model
############################################

def load_sample_image(image_name='example.jpg'):
    """
    Load and preprocess a sample image. If not available, generate a constant dummy image.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # Get the absolute path to the "images" directory
    images_dir = os.path.join(os.path.dirname(__file__), "images")

    # Construct the full image path
    image_path = os.path.join(images_dir, image_name)    
    if os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        print(f"Loaded image from {image_path}")
    else:
        print("Image not found. Generating a constant dummy image.")
        dummy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
        img = Image.fromarray(dummy_img)
    img_tensor = transform(img)
    return img_tensor

def adversarial_test(model, img_tensor):
    """Run an adversarial attack (FGSM) on the model using ART."""
    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=(3, 224, 224),
        nb_classes=1000,
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )
    img_np = img_tensor.unsqueeze(0).numpy()
    preds_orig = classifier.predict(img_np)
    orig_class = int(np.argmax(preds_orig))
    attack = FastGradientMethod(estimator=classifier, eps=0.05)
    adv_examples = attack.generate(x=img_np)
    preds_adv = classifier.predict(adv_examples)
    adv_class = int(np.argmax(preds_adv))
    print(f"Adversarial Test: {orig_class} -> {adv_class}")
    return orig_class, adv_class

def bias_analysis_test():
    """Perform bias analysis using fixed predictions and subgroup labels."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 0])
    subgroups = np.array(['group_a'] * 5 + ['group_b'] * 5)
    metrics = MetricFrame(metrics={
        'accuracy': lambda yt, yp: np.mean(yt == yp),
        'selection_rate': selection_rate,
        'TPR': true_positive_rate,
        'FPR': false_positive_rate,
    }, y_true=y_true, y_pred=y_pred, sensitive_features=subgroups)
    print("\nBias Analysis Metrics by Group:")
    print(metrics.by_group)
    return metrics

def security_test(model, img_tensor):
    """Test the model's robustness to a fixed input perturbation."""
    noisy_input = img_tensor + torch.full_like(img_tensor, 0.1)
    with torch.no_grad():
        output_clean = model(img_tensor.unsqueeze(0))
        output_noisy = model(noisy_input.unsqueeze(0))
    diff = torch.abs(output_clean - output_noisy).mean().item()
    print(f"Security Test: Difference = {diff:.4f}")
    return diff

def performance_test(model, img_tensor, iterations=10):
    """Measure average inference time over a number of iterations."""
    times = []
    with torch.no_grad():
        _ = model(img_tensor.unsqueeze(0))  # Warm-up
    for _ in range(iterations):
        start = time.time()
        with torch.no_grad():
            _ = model(img_tensor.unsqueeze(0))
        end = time.time()
        times.append(end - start)
    avg_time = np.mean(times)
    print(f"Performance Test: Avg inference time = {avg_time:.4f} sec")
    return avg_time

def confidence_test(model, img_tensor):
    """Compute the maximum softmax probability (confidence) from the model's prediction."""
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
    probabilities = F.softmax(output, dim=1)
    max_confidence = probabilities.max().item()
    print(f"Confidence Test: Max softmax probability = {max_confidence:.4f}")
    return max_confidence

# --- Extra tests ---

def gradient_norm_test(model, img_tensor):
    """Measure the norm of the gradient with respect to the input."""
    input_tensor = img_tensor.unsqueeze(0).clone().detach().requires_grad_(True)
    output = model(input_tensor)
    output_sum = output.sum()
    output_sum.backward()
    grad_norm = input_tensor.grad.norm().item()
    print(f"Gradient Norm Test: {grad_norm:.4f}")
    return grad_norm

def activation_sparsity_test(model, img_tensor, threshold=0.1):
    """Compute the fraction of activations in the first Conv2d layer that are below a threshold."""
    activations = []
    def hook(module, input, output):
        activations.append(output.detach())
    first_conv = None
    for child in model.children():
        if isinstance(child, torch.nn.Conv2d):
            first_conv = child
            break
    if first_conv is None:
        print("Activation Sparsity Test: No Conv2d layer found.")
        return 0.0
    handle = first_conv.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(img_tensor.unsqueeze(0))
    handle.remove()
    if activations:
        act = activations[0]
        sparsity = (act.abs() < threshold).float().mean().item()
        print(f"Activation Sparsity Test: {sparsity:.4f}")
        return sparsity
    return 0.0

def parameter_count_test(model):
    """Count the total number of parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter Count Test: {total_params}")
    return total_params

def memory_usage_test(model, img_tensor):
    """
    Estimate the memory usage (in MB) during inference.
    If GPU is available, uses CUDA peak memory; otherwise, approximates based on parameter count.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        img_tensor = img_tensor.to(device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(img_tensor.unsqueeze(0))
        mem_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Memory Usage Test (GPU): {mem_usage:.4f} MB")
        return mem_usage
    else:
        total_params = sum(p.numel() for p in model.parameters())
        mem_usage = total_params * 4 / (1024 ** 2)  # approximate in MB assuming float32 (4 bytes)
        print(f"Memory Usage Test (CPU approx): {mem_usage:.4f} MB")
        return mem_usage

def occlusion_test(model, img_tensor, occlusion_size=50):
    """
    Occlude a central patch of the image and measure the drop in confidence.
    """
    img_occluded = img_tensor.clone()
    _, h, w = img_occluded.shape
    start_h = h // 2 - occlusion_size // 2
    start_w = w // 2 - occlusion_size // 2
    img_occluded[:, start_h:start_h+occlusion_size, start_w:start_w+occlusion_size] = 0
    with torch.no_grad():
        output_orig = model(img_tensor.unsqueeze(0))
        output_occ = model(img_occluded.unsqueeze(0))
    prob_orig = F.softmax(output_orig, dim=1).max().item()
    prob_occ = F.softmax(output_occ, dim=1).max().item()
    drop = prob_orig - prob_occ
    print(f"Occlusion Test: Confidence drop = {drop:.4f}")
    return drop

def compute_overall_score(adv_result, bias_metrics, security_result):
    """Compute a simple overall score based on three tests for grading."""
    adv_penalty = 0 if adv_result[0] == adv_result[1] else 1
    bias_diff = abs(bias_metrics.by_group.loc['group_a']['accuracy'] - 
                    bias_metrics.by_group.loc['group_b']['accuracy'])
    if bias_diff < 0.05:
        bias_penalty = 0
    elif bias_diff < 0.1:
        bias_penalty = 1
    else:
        bias_penalty = 2
    if security_result < 0.1:
        sec_penalty = 0
    elif security_result < 0.2:
        sec_penalty = 1
    else:
        sec_penalty = 2
    total_penalty = adv_penalty + bias_penalty + sec_penalty
    if total_penalty == 0:
        grade = "A (Top)"
    elif total_penalty == 1:
        grade = "B (Good)"
    elif total_penalty == 2:
        grade = "C (OK)"
    elif total_penalty == 3:
        grade = "D (Bad)"
    else:
        grade = "F (Trash)"
    return grade, total_penalty, adv_penalty, bias_penalty, sec_penalty, bias_diff

############################################
# Endpoints
############################################

@app.route('/model/<model_name>')
def one_model(model_name):
    """Endpoint for a single model."""
    try:
        model = download_and_save_model(model_name)
    except ValueError as e:
        return f"<h1>Error: {e}</h1>"
    img_tensor = load_sample_image()
    adv_result = adversarial_test(model, img_tensor)
    bias_metrics = bias_analysis_test()
    security_result = security_test(model, img_tensor)
    perf_time = performance_test(model, img_tensor, iterations=10)
    confidence = confidence_test(model, img_tensor)
    # Extra tests
    grad_norm = gradient_norm_test(model, img_tensor)
    sparsity = activation_sparsity_test(model, img_tensor)
    param_count = parameter_count_test(model)
    mem_usage = memory_usage_test(model, img_tensor)
    occ_drop = occlusion_test(model, img_tensor)
    grade, total_penalty, adv_pen, bias_pen, sec_pen, bias_diff = compute_overall_score(adv_result, bias_metrics, security_result)
    
    result = {
        "model_name": model_name,
        "adversarial": f"{adv_result[0]} -> {adv_result[1]}",
        "bias_diff": f"{bias_diff:.4f}",
        "security": f"{security_result:.4f}",
        "perf_time": f"{perf_time:.4f} sec",
        "confidence": f"{confidence:.4f}",
        "grad_norm": f"{grad_norm:.4f}",
        "sparsity": f"{sparsity:.4f}",
        "param_count": param_count,
        "mem_usage": f"{mem_usage:.4f} MB",
        "occlusion_drop": f"{occ_drop:.4f}",
        "grade": grade,
        "total_penalty": total_penalty
    }
    return render_template("model.html", result=result)

@app.route('/all')
def all_models():
    """Endpoint for testing all available models."""
    results_list = []
    for model_name in MODEL_LOADERS.keys():
        try:
            model = download_and_save_model(model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue
        img_tensor = load_sample_image()
        adv_result = adversarial_test(model, img_tensor)
        bias_metrics = bias_analysis_test()
        security_result = security_test(model, img_tensor)
        perf_time = performance_test(model, img_tensor, iterations=10)
        confidence = confidence_test(model, img_tensor)
        # Extra tests
        grad_norm = gradient_norm_test(model, img_tensor)
        sparsity = activation_sparsity_test(model, img_tensor)
        param_count = parameter_count_test(model)
        mem_usage = memory_usage_test(model, img_tensor)
        occ_drop = occlusion_test(model, img_tensor)
        grade, total_penalty, adv_pen, bias_pen, sec_pen, bias_diff = compute_overall_score(adv_result, bias_metrics, security_result)
        result = {
            "model_name": model_name,
            "adversarial": f"{adv_result[0]} -> {adv_result[1]}",
            "bias_diff": f"{bias_diff:.4f}",
            "security": f"{security_result:.4f}",
            "perf_time": f"{perf_time:.4f} sec",
            "confidence": f"{confidence:.4f}",
            "grad_norm": f"{grad_norm:.4f}",
            "sparsity": f"{sparsity:.4f}",
            "param_count": param_count,
            "mem_usage": f"{mem_usage:.4f} MB",
            "occlusion_drop": f"{occ_drop:.4f}",
            "grade": grade,
            "total_penalty": total_penalty
        }
        results_list.append(result)
    return render_template("all.html", results=results_list)

@app.route('/multiple/<models_str>')
def multiple_models(models_str):
    """
    Endpoint for testing multiple models provided as a comma-separated list.
    Example: /multiple/resnet50,mobilenet_v2
    """
    model_names = [name.strip() for name in models_str.split(',')]
    results_list = []
    for model_name in model_names:
        try:
            model = download_and_save_model(model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue
        img_tensor = load_sample_image()
        adv_result = adversarial_test(model, img_tensor)
        bias_metrics = bias_analysis_test()
        security_result = security_test(model, img_tensor)
        perf_time = performance_test(model, img_tensor, iterations=10)
        confidence = confidence_test(model, img_tensor)
        # Extra tests
        grad_norm = gradient_norm_test(model, img_tensor)
        sparsity = activation_sparsity_test(model, img_tensor)
        param_count = parameter_count_test(model)
        mem_usage = memory_usage_test(model, img_tensor)
        occ_drop = occlusion_test(model, img_tensor)
        grade, total_penalty, adv_pen, bias_pen, sec_pen, bias_diff = compute_overall_score(adv_result, bias_metrics, security_result)
        result = {
            "model_name": model_name,
            "adversarial": f"{adv_result[0]} -> {adv_result[1]}",
            "bias_diff": f"{bias_diff:.4f}",
            "security": f"{security_result:.4f}",
            "perf_time": f"{perf_time:.4f} sec",
            "confidence": f"{confidence:.4f}",
            "grad_norm": f"{grad_norm:.4f}",
            "sparsity": f"{sparsity:.4f}",
            "param_count": param_count,
            "mem_usage": f"{mem_usage:.4f} MB",
            "occlusion_drop": f"{occ_drop:.4f}",
            "grade": grade,
            "total_penalty": total_penalty
        }
        results_list.append(result)
    return render_template("all.html", results=results_list)

if __name__ == '__main__':
    app.run(debug=True)
