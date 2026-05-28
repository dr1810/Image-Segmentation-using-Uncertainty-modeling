### Model Architectures

The project evaluates three separate encoder-decoder paradigms leveraging transfer learning:

#### 1. ResNet50-Based Segmentation Model
This architecture adapts a standard deep image classification backbone for a semantic segmentation task.
* **Encoder (Backbone):** A pre-trained `ResNet50` network with its weights frozen (`layer.trainable = False`) to serve as a fixed feature extractor.
* **Segmentation Head:** A $1 \times 1$ Convolutional layer (`Conv2D(3, (1,1))`) that maps the extracted core features directly down to 3 channels using a `sigmoid` activation function.
* **Decoder / Upscaling:** A large $32 \times 32$ `UpSampling2D` block is appended at the very end to scale the low-resolution bottleneck feature maps back up to match the native input resolution ($224 \times 224 \times 3$).
* **Compilation:** Optimized using the `Adam` optimizer and a standard `Binary Crossentropy` loss function.

#### 2. InceptionV3-Based Segmentation Model
This model utilizes a more progressive, deep decoder approach paired with a different structural backbone.
* **Encoder (Backbone):** A pre-trained `InceptionV3` network acting as the feature encoder, designed to process input shapes of $299 \times 299 \times 3$.
* **Progressive Segmentation Head:** Instead of a single upscaling jump, it implements a step-by-step decoding block:
  - `Conv2D(256, (3,3), activation='relu')`
  - `Conv2D(128, (3,3), activation='relu')`
  - `UpSampling2D((2,2))`
  - `Conv2D(64, (3,3), padding='same', activation='relu')`
  - `UpSampling2D((2,2))`
  - `Conv2D(3, (3,3), padding='same', activation='sigmoid')`
* **Resolution Bottleneck Constraint:** Because of the structural dimensions of this custom head, the original ground-truth annotation targets are dynamically resized down to a low-resolution map of $16 \times 16$ pixels during the training loop execution to align with the core bottleneck geometry.

#### 3. EfficientNetB7-Based Segmentation Model
A high-capacity configuration designed to benchmark the capabilities of a modern, heavily scaled scaling backbone.
* **Encoder (Backbone):** A massive `EfficientNetB7` base network utilizing pre-trained ImageNet features, receiving input shapes scaled to $224 \times 224 \times 3$.
* **Decoder Structure:** To provide a direct baseline comparison against the first model, it mirrors the structural strategy of the ResNet50 model—employing a single $1 \times 1$ Convolutional layer coupled with a massive $32 \times 32$ bilinear/nearest-neighbor upsampling block to output a 3-channel semantic mask.

---

### Advanced Ensembling & Uncertainty Models

To go beyond standalone networks, the individual backbones were fused together into custom meta-frameworks designed to output predictions alongside calculated **epistemic uncertainty** (model confidence variance).

#### A. Bayesian Averaging Model (PyTorch Implementation)
A custom-engineered PyTorch module wrapper that takes two separate upstream model outputs and performs a simultaneous estimation of predictive mean and localized boundary variance:
* **Mean Output (Prediction):** Computed as the strict arithmetic mean between both systems:  
  $$\text{Mean} = \frac{\text{Output}_1 + \text{Output}_2}{2}$$
* **Uncertainty Quantification:** Captures localized predictive disagreement across boundaries by measuring the absolute differences between their predictions:  
  $$\text{Uncertainty} = \frac{|\text{Output}_1 - \text{Output}_2|}{2}$$

#### B. Uncertainty-Weighted Ensemble Model (TensorFlow Implementation)
An advanced meta-model inheriting directly from `tf.keras.Model` that dynamically fuses two base networks (such as the base model and the EfficientNet model). Instead of basic averaging, it performs **Inverse Variance Weighting**:
* **Dynamic Penalization:** It computes continuous operational weights that are inversely proportional to each individual network's predictive variance ($\sigma$). If a model is highly uncertain about a specific pixel region, its vote weight is automatically scaled down:  
  $$W_1 = \frac{1}{\sigma_1}, \quad W_2 = \frac{1}{\sigma_2}$$
* **Fused Predictive Mean:** Fuses predictions using normalized confidence weights:  
  $$\text{Mean Output} = \frac{W_1 \cdot \text{Output}_1 + W_2 \cdot \text{Output}_2}{W_1 + W_2}$$
* **Combined System Uncertainty:** Computes the total residual predictive uncertainty propagating through the ensemble using the joint weight scaling factor:  
  $$\text{Combined Uncertainty} = \sqrt{W_1 \cdot \sigma_1^2 + W_2 \cdot \sigma_2^2}$$
