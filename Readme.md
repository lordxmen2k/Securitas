# SECURITAS: Security Evaluation and Control Utilizing Reliable Intelligent Transparent Automated Systems

**Author:**  
Gerald Enrique Nelson Mc Kenzie

## **Abstract**
This publication presents an open-source framework designed to rigorously evaluate artificial intelligence (AI) models using a suite of quantitative tests, all integrated into a Flask-based web application. The framework provides multiple endpoints for diverse evaluation needs:
- **`/model/<model_name>`** for testing a single model.
- **`/all`** for evaluating all available models.
- **`/multiple/<models_str>`** for assessing a comma-separated list of models.

The evaluation suite comprises a broad range of tests that address critical aspects of model performance and robustness:
- **Adversarial Test (FGSM Attack):** Evaluates model susceptibility to adversarial perturbations by comparing predictions before and after generating adversarial examples.
- **Bias Analysis Test:** Assesses fairness by comparing performance across synthetic subgroups.
- **Security Test (Input Perturbation):** Measures the change in model output under fixed input noise.
- **Performance Test (Inference Time):** Computes average inference time to determine real-time applicability.
- **Confidence Test (Max Softmax Probability):** Gauges the model's prediction confidence.
- **Gradient Norm Test:** Quantifies sensitivity to input variations by calculating the norm of the gradient.
- **Activation Sparsity Test:** Estimates network efficiency by measuring the sparsity of activations in early convolutional layers.
- **Parameter Count Test:** Counts the total number of trainable parameters as a measure of model complexity.
- **Memory Usage Test:** Estimates the memory footprint during inference, crucial for deployment.
- **Occlusion Test:** Determines reliance on input regions by measuring confidence drop when a central patch is occluded.

Each test is accompanied by a rating and a detailed explanation, offering a comprehensive perspective on the model's robustness, efficiency, and reliability. This framework not only serves as a proof-of-concept for standardized AI model evaluation but also aims to contribute to the development of safe and transparent AI systems, informing best practices and regulatory standards.

The application is intended for researchers, practitioners, and policymakers who require a robust baseline for assessing AI models, with potential applications in national security, healthcare, finance, and beyond.

## **Overview**

SECURITAS aims to simplify the process of evaluating AI models securely by automating key security and monitoring aspects. This project includes:
- **Adversarial Robustness:** Tests models against adversarial perturbations to assess vulnerability.
- **Bias Auditing:** Conducts fairness analysis to identify potential model biases across synthetic subgroups.
- **Security Testing:** Evaluates model response to input perturbations to detect vulnerabilities.
- **Performance Monitoring:** Computes real-time inference time and measures computational efficiency.
- **Compliance Logging:** Tracks governance, audit logs, and regulatory compliance to ensure transparency and accountability.

## **Repository Structure**

- **`securitas_monitor.py`**: Contains the main implementation of the `SecuritasMonitor` class that executes model evaluation tests.
- **`requirements.txt`**: Lists the Python dependencies required to run the code.
- **`README.md`**: This file, which provides an overview and usage instructions.

## **Installation**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/lordxmen2k/SECURITAS.git
   cd SECURITAS
   ```

2. **Create a Conda Virtual Environment:**
    ```bash
    conda create -n securitas_env python=3.9
    ```

3. **Activate the Conda Environment:**
    ```bash
    conda activate securitas_env
    ```

4. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## **Usage**

1. SECURITAS provides an API to test AI models for security and robustness. To run the application:
   ```bash
   python app.py
   ```

2. Upon execution, the script will:

   - Initialize the secure evaluation environment.
   - Execute performance, security, and bias tests on the given AI model.
   - Monitor inference time and memory usage.
   - Generate an evaluation report summarizing security, bias, and performance metrics.

## **Extending SECURITAS**

- **Adding Custom Tests:** Extend the existing evaluation suite to include more domain-specific security and fairness tests.
- **Enhancing Compliance Monitoring:** Integrate blockchain-based audit trails for tamper-proof compliance verification.
- **Scaling for Enterprise Use:** Deploy SECURITAS as a scalable API service for large-scale AI model evaluations.

## **License & Citation**

This project is open-source under the MIT License. If you use this framework in your research, please cite:

@article{Securitas2025,
  author = {Gerald Enrique Nelson Mc Kenzie},
  title = {SECURITAS: Security Evaluation and Control Utilizing Reliable Intelligent Transparent Automated Systems},
  journal = {Zenodo},
  year = {2025},
  doi = {10.5281/zenodo.14878539},
  url = {https://github.com/lordxmen2k/Securitas}
}