
# Deep Learning for Medical Imaging
The following projects were implemented as part of the course "Deep Learning for Medical Imaging" at the VU Amsterdam. 

## 📁 Assignments Overview

1. **Assignment 1** – IVIM Parameter Estimation using PyTorch  
   Implement a neural network from scratch to estimate the perfusion fraction `f` from diffusion-weighted MRI signals based on the IVIM model.
2. **Assignment 2** – Skin Lesion Classification and Regression (ISIC 2019 Challenge)  
   In this assignment, we implemented two deep learning models for classifying dermoscopic images from the ISIC 2019 dataset: a **custom convolutional neural network (CustomConvNet)** and a **transfer learning model based on ResNet50 (TransConvNet)**.

   For the CustomConvNet, we improved upon a baseline by:
   - Increasing depth to five convolutional layers (32→512 filters)
   - Adding a **skip connection** to improve gradient flow
   - Replacing ReLU with **LeakyReLU** to avoid dead neurons
   - Incorporating **global average pooling**, dropout, and dense layers for better regularization and classification

   For the transfer learning approach, we fine-tuned a **pretrained ResNet50** by:
   - Replacing the final fully connected layer with a custom classifier
   - Freezing early layers while allowing the last residual block to be trainable
   - Using **data augmentation** (flips and Gaussian blur) to improve generalization

   To prioritize recall and penalize false negatives, we used **Focal Tversky Loss** in both models, which is especially suitable for medical data where missing a diagnosis is critical.

   For segmentation, we also implemented a **modified U-Net** architecture:
   - We added **padding** to preserve spatial dimensions
   - Applied **1×1 convolutional layers** in skip connections to refine encoder features
   - Used **LeakyReLU** instead of ReLU and added **batch normalization** for stable training
   - Included **dropout** in the bottleneck for regularization

   Among the different loss functions tested (Dice, Focal Tversky, BCE), **Binary Cross-Entropy (BCE)** yielded the best balance of stability and segmentation accuracy.

   **Results:**  
   - TransConvNet achieved the highest overall performance, with better **accuracy and F1-score**
   - CustomConvNet improved recall over the baseline and showed strong robustness with the new architecture and loss
   - U-Net achieved a final validation accuracy of ~0.94 and F1-score around 0.85, with steadily improving precision and recall throughout training

   Overall, our improvements in architecture, training strategy, and loss functions helped both classification and segmentation models generalize well to unseen medical images.

3. **Assignment 3** – Exploring the Impact of K-Space Interpolation and Masking Acceleration on MRI Reconstruction Using VarNet  
   In this assignment, we investigated how different **k-space interpolation methods** impact the performance of **VarNet**, a deep-learning-based MRI reconstruction network. MRI data is naturally acquired in the frequency domain (k-space), and to accelerate acquisition, it is common practice to **undersample** this space using masks with **acceleration factors** (AF) such as 4, 6, and 8. Instead of relying solely on VarNet to recover missing data, we explored **pre-filling missing k-space values using interpolation** before feeding the data into the network.

   We tested the following interpolation techniques:
   - **Nearest Neighbor (NN)**  
   - **Cubic Spline (B-spline)**  
   - **Fourier Interpolation**  
   - **Radial Basis Function (RBF)**  

   **Pipeline Overview:**  
   1. Load fully sampled k-space and image data  
   2. Apply undersampling mask (AF = 4, 6, 8)  
   3. Interpolate missing k-space values using one of the above methods  
   4. Estimate coil sensitivity maps  
   5. Feed pre-processed data into VarNet  
   6. Generate reconstructed MRI image  
   7. Evaluate performance using both **quantitative and qualitative metrics**  

   **Quantitative Evaluation Metrics:**  
   - **PSNR (Peak Signal-to-Noise Ratio):** Measures reconstruction fidelity in terms of peak vs. noise  
   - **NMSE (Normalized Mean Squared Error):** Captures relative reconstruction error  
   - **SSIM (Structural Similarity Index):** Assesses perceptual similarity in terms of luminance, contrast, and structure  

   **Qualitative Evaluation:**  
   We also performed visual comparisons of the reconstructed images with ground truth magnitude images to assess how well structural details were preserved.

   **Results & Observations:**  
   - **Fourier interpolation** consistently provided the best initial reconstructions, preserving the frequency domain’s structure and symmetry.
   - **Nearest Neighbor** interpolation performed the worst due to its simplicity, failing to reconstruct meaningful values in regions with sparse samples.
   - **B-spline interpolation** introduced ghosting and shadow artifacts due to oversmoothing, particularly at higher acceleration factors.
   - **RBF interpolation** offered a balanced performance, though it struggled with sharp frequency transitions.

   Interestingly, despite clear differences in the initial interpolated k-space, **all final reconstructions became visually and quantitatively similar after VarNet processing**, showcasing the model’s **robust learned regularization and data consistency** steps. This suggests that VarNet is able to correct for imperfections introduced by different interpolation methods through its iterative refinement process.

   Overall, our results highlight the power of VarNet in MRI reconstruction and suggest that while interpolation may slightly influence early representations, the network compensates effectively across cascades, especially at lower acceleration factors.




