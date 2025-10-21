import os
import rasterio
import numpy as np

def create_siamese_patches(before_path, after_path, mask_path, out_path,
                           patch_size= 16, samples_per_class=1000):
    """Create before/after patch pairs with masks for Siamese CNN training. 
    Set at 16 x 16 image size and 1000 sample per class to ensure collection of all data. """

    # Load input raster files
    with rasterio.open(before_path) as src_b, rasterio.open(after_path) as src_a, rasterio.open(mask_path) as src_m:
        before_img, after_img, mask = src_b.read(), src_a.read(), src_m.read(1)

    H, W = mask.shape
    damaged, undamaged= [], []

    # Random sampling loop made until both sets are filled 
    # This is set to loop four times and stops for each sample which hits 1000, if it does not then the loop will collect the max amount
    # The loop is looped four times to ensure that there are 4000 attempts of finding damaged patches, to reduce missing any damaged patches.
    for _ in range(samples_per_class * 4):
        x, y = np.random.randint(0, W - patch_size), np.random.randint(0, H - patch_size)
        before_patch = before_img[:, y:y+patch_size, x:x+patch_size]
        after_patch = after_img[:, y:y+patch_size, x:x+patch_size]
        mask_patch = mask[y:y+patch_size, x:x+patch_size]

        if before_patch.shape[1:] != (patch_size, patch_size):
            continue  # skip incomplete edge patches

        target = damaged if mask_patch.sum() > 0 else undamaged
        if len(target) < samples_per_class:
            target.append((before_patch, after_patch, mask_patch))

        if len(damaged) >= samples_per_class and len(undamaged) >= samples_per_class:
            break

    # Stack results into arrays
    before_array = np.array([p[0] for p in damaged + undamaged])
    after_array = np.array([p[1] for p in damaged + undamaged])
    masks_array = np.array([p[2] for p in damaged + undamaged])

    # Save dataset
    np.savez_compressed(out_path, before=before_array, after=after_array, masks=masks_array)
    print(f"saved {masks_array.shape[0]} pairs to {out_path}")


base_dir = r"..." # insert root of the TIF before/after and mask files
out_dir = os.path.join(base_dir, "patches_siamese")
os.makedirs(out_dir, exist_ok=True)  # make sure folder exists

# File paths
# Each entry specifies the before image, after image, and mask file for a city, example of how they are listed in the placeholder below
cities = [
    {
        "name": "region name",
        "before": r"",
        "after": r"",
        "mask": r""
    },
]

# Run patches for the cities
for city in cities:
    out_path = os.path.join(out_dir, f"{city['name']}_pairs.npz")
    create_siamese_patches(city["before"], city["after"], city["mask"], out_path)
