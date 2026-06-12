"""
Color Extractor Module
Extracts base colors from environment images using K-means clustering
"""
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2


class ColorExtractor:
    """
    Extracts dominant colors from environment images using K-means clustering.
    As described in the paper: base colors c = {c1, c2, c3} are extracted
    by partitioning all pixels into distinct categories.
    """

    def __init__(self, num_colors=3, max_iter=100):
        """
        Args:
            num_colors: Number of base colors to extract (k in K-means)
            max_iter: Maximum iterations for K-means
        """
        self.num_colors = num_colors 
        self.max_iter = max_iter

    def extract_from_image(self, image):
        """
        Extract base colors from a single image.

        Args:
            image: PIL Image or numpy array (H, W, 3) in RGB format

        Returns:
            base_colors: numpy array of shape (num_colors, 3) containing RGB values
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Reshape image to (N, 3) where N is number of pixels
        pixels = image.reshape(-1, 3).astype(np.float32)

        # Apply K-means clustering
        kmeans = KMeans(
            n_clusters=self.num_colors,
            max_iter=self.max_iter,
            random_state=42,
            n_init=10
        )
        kmeans.fit(pixels)

        # Get cluster centers as base colors
        base_colors = kmeans.cluster_centers_

        # Sort by cluster size (most frequent color first)
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        sorted_indices = np.argsort(-counts)
        base_colors = base_colors[sorted_indices]

        return base_colors.astype(np.float32)

    def extract_from_images(self, images):
        """
        Extract base colors from multiple images with similar environment.

        Args:
            images: List of PIL Images or numpy arrays

        Returns:
            base_colors: numpy array of shape (num_colors, 3)
        """
        all_pixels = []

        for image in images:
            if isinstance(image, Image.Image):
                image = np.array(image)
            pixels = image.reshape(-1, 3).astype(np.float32)
            # Subsample for efficiency
            if len(pixels) > 10000:
                indices = np.random.choice(len(pixels), 10000, replace=False)
                pixels = pixels[indices]
            all_pixels.append(pixels)

        all_pixels = np.vstack(all_pixels)

        # Apply K-means clustering
        kmeans = KMeans(
            n_clusters=self.num_colors,
            max_iter=self.max_iter,
            random_state=42,
            n_init=10
        )
        kmeans.fit(all_pixels)

        # Get cluster centers as base colors
        base_colors = kmeans.cluster_centers_

        # Sort by cluster size
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        sorted_indices = np.argsort(-counts)
        base_colors = base_colors[sorted_indices]

        return base_colors.astype(np.float32)

    def visualize_colors(self, base_colors, save_path=None):
        """
        Visualize extracted base colors.

        Args:
            base_colors: numpy array of shape (num_colors, 3)
            save_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 2))

        for i, color in enumerate(base_colors):
            normalized_color = color / 255.0
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=normalized_color))

        ax.set_xlim(0, len(base_colors))
        ax.set_ylim(0, 1)
        ax.set_xticks([i + 0.5 for i in range(len(base_colors))])
        ax.set_xticklabels([f'Color {i+1}' for i in range(len(base_colors))])
        ax.set_yticks([])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Test the color extractor
    extractor = ColorExtractor(num_colors=3)

    # Create a sample image
    sample_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Extract colors
    colors = extractor.extract_from_image(sample_image)
    print(f"Extracted base colors:\n{colors}")
