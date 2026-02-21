"""Visualize prediction heatmap over image 
"""

from liver_fibrosis_detection import plt, cv2, np

# ============================================================================
# 5. ENHANCED HEATMAP VISUALIZATION
# ============================================================================

def visualize_prediction(img_pil, heatmap, probability, prediction):
    """
    Professional visualization with colored overlay
    """
    plt.figure(figsize=(14, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img_pil)
    plt.title("Hepatic Ultrasound", fontsize=12, fontweight='bold')
    plt.axis('off')

    # Heatmap alone
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Activation Map (GradCAM)", fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # Overlay
    plt.subplot(1, 3, 3)
    img_resized = img_pil.resize((224, 224))
    img_array = np.array(img_resized)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (heatmap_colored * 0.5 + img_array * 0.5).astype(np.uint8)

    plt.imshow(overlay)

    # Title with color based on prediction
    color = 'red' if prediction == "FIBROSIS SUSPICION" else 'green'
    plt.title(f"{prediction}\n(Score: {probability*100:.1f}%)",
              fontsize=12, fontweight='bold', color=color)
    plt.axis('off')

    plt.tight_layout()
    plt.show()