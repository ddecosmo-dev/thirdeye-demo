Step 1: 
Get initial model running on windows
complete

Step 2
Add prefilter to this model, should also be included in output text

# This section now runs the fast_quality_prefilter on the dataset
# Note: Since the dataset is in memory as PIL images, we adapt the filter logic

def run_prefilter_on_dataset(hf_dataset, index, blur_thresh=100):
    # Convert PIL to OpenCv format (Grayscale)
    pil_img = hf_dataset[index]['image']
    cv_img = np.array(pil_img.convert('L'))

    # 1. Lighting Check
    mean_intensity = np.mean(cv_img)
    if mean_intensity < 20: return False, "Too Dark"
    if mean_intensity > 235: return False, "Overexposed"

    # 2. Blur Check
    laplacian_var = cv2.Laplacian(cv_img, cv2.CV_64F).var()
    if laplacian_var < blur_thresh: return False, "Too Blurry"

    return True, "Passed"

    