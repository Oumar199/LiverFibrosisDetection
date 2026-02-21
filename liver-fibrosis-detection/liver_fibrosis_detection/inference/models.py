from liver_fibrosis_detection import nn, models, torch, F, cv2

# ============================================================================
# 2. ARCHITECTURE MODÈLE V4.1 (Compatible avec l'entraînement)
# ============================================================================

class HybridModelV4_1(nn.Module):
    """
    Architecture EXACTEMENT identique à training_balanced.py
    """
    def __init__(self, num_clinical):
        super().__init__()

        # CNN (EfficientNet-B0)
        self.cnn = models.efficientnet_b0(weights=None)
        cnn_out = self.cnn.classifier[1].in_features  # 1280
        self.cnn.classifier = nn.Identity()

        self.flatten = nn.Flatten()

        # MLP (Données Cliniques) - Dropout 0.6
        self.mlp = nn.Sequential(
            nn.Linear(num_clinical, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Fusion - Dropout 0.6
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, img, clin):
        # CNN
        x1 = self.cnn(img)
        x1 = self.flatten(x1)

        # MLP
        x2 = self.mlp(clin)

        # Fusion
        concat = torch.cat((x1, x2), dim=1)
        logits = self.fusion(concat)

        return logits


# ============================================================================
# 3. GRADCAM AMÉLIORÉ (Pour Heatmap)
# ============================================================================

class GradCAMV4_1:
    """
    GradCAM optimisé pour EfficientNet-B0
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook sur dernière couche CNN (avant avgpool)
        target_layer = self.model.cnn.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img, clin, target_class=1):
        """
        Génère la heatmap pour la classe cible (1 = malade)
        """
        self.model.zero_grad()

        # Forward
        logits = self.model(img, clin)
        probs = F.softmax(logits, dim=1)

        # Backward sur classe cible
        score = logits[0, target_class]
        score.backward()

        # Calcul heatmap
        if self.gradients is None or self.activations is None:
            return None, probs[0, target_class].item()

        # Moyenne des gradients (importance de chaque canal)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Somme pondérée des activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]

        # ReLU (ne garder que les activations positives)
        cam = F.relu(cam)

        # Normalisation [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Resize 224x224
        cam = cv2.resize(cam, (224, 224))

        return cam, probs[0, target_class].item()
