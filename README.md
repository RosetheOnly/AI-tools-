# AI-tools-
Q1: Primary differences between TensorFlow and PyTorch, and when to choose one over the other.
1.Computation Graph
TensorFlow:

Static graph (define-then-run) by default.

Supports dynamic graphs via eager execution.

PyTorch:

Dynamic graph (define-by-run) by default.

More intuitive for debugging and iterative development.
2.API Design
TensorFlow:

Multi-layered API (low-level TF ops + high-level Keras).

Steeper learning curve.

PyTorch:

Pythonic, object-oriented design.

Simpler and more intuitive for Python developers.

3.Deployment
TensorFlow:

Production-ready tools (TF Serving, TF Lite, TF.js).

Superior for mobile/edge deployment.PyTorch:

Relies on TorchServe/ONNX for deployment (rapidly improving).

Traditionally stronger in research than production.

4.Community & Ecosystem
TensorFlow:

Industry-dominated (Google, large-scale systems).

Extensive production tooling (TFX, Kubeflow).

PyTorch:

Research-dominated (academia, latest papers).
5.Debugging & Visualization
TensorFlow:

TensorBoard (advanced tracking, profiling).

PyTorch:

Supports TensorBoard + lightweight alternatives (e.g., Weights & Biases).

Easier debugging due to dynamic graphs.
Preferred for rapid prototyping and state-of-the-art models.


When to Choose One Over the Other
Choose TensorFlow if:

Deploying models to production (web/mobile/edge),

Building end-to-end ML pipelines (TFX),

Working with large-scale distributed systems.

Example: Deploying a model to Android via TF Lite.Choose PyTorch if:

Rapid research prototyping or academia,

Prioritizing debugging flexibility (dynamic graphs),

Leveraging latest research models (e.g., Hugging Face transformers).

Example: Experimenting with a novel neural architecture.
Q2: Two use cases for Jupyter Notebooks in AI development
1.Interactive Prototyping & Debugging:

.Execute code incrementally (cell-by-cell) to test model components (e.g., data preprocessing, layer outputs).

.Visualize results immediately (e.g., plot loss curves, display sample predictions) without rerunning entire scripts.

.Example: Debugging a CNN by inspecting feature maps after each convolutional layer.

2.Collaborative Documentation & Reproducibility:

.Combine code, visualizations, equations (LaTeX), and narrative text (Markdown) in a single shareable document.

.Share notebooks (via .ipynb files or platforms like Google Colab) to ensure experiments are reproducible.

.Example: Documenting an NLP pipeline—data cleaning, model training, and results—for team review.

Q3: How spaCy enhances NLP tasks vs. basic Python string operations
1.Linguistic Intelligence:

.spaCy uses statistical models to understand context (e.g., "Apple" as company vs. fruit), while string operations rely on rigid rules.

.Example: spaCy identifies entities (ORG, PERSON) in text; string ops would need error-prone regex patterns.

2.Efficiency & Scalability:

.Built-in tokenization, lemmatization, and dependency parsing handle complex language structures at high speed (Cython-optimized).

.String operations (e.g., .split(), regex) struggle with edge cases (hyphenated words, multilingual text).

3.Pre-trained Pipelines:

.Offers ready-to-use models (e.g., en_core_web_sm) for tasks like named entity recognition (NER), part-of-speech tagging, or similarity detection.

.String operations require manual implementation for each task.




