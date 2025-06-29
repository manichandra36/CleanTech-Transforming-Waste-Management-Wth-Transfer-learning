import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base_dir = "dataset"
classes = ["biodegradable", "recyclable", "trash"]

for category in classes:
    folder_path = os.path.join(base_dir, category)
    num_images = len(os.listdir(folder_path))
    print(f"{category}: {num_images} images")

    # Show 3 sample images
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    fig.suptitle(f"Sample Images - {category}")

    for i, img_name in enumerate(os.listdir(folder_path)[:3]):
        img_path = os.path.join(folder_path, img_name)
        img = mpimg.imread(img_path)
        axes[i].imshow(img)
        axes[i].axis("off")
    plt.show()
