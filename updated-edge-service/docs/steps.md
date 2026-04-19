Step 1: 
Get initial model running on windows
complete

Step 2
Add prefilter to this model, should also be included in output text
images should be before the prefilter, but should not be imageNet normalized like inputs to NN node

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

COMPLETED! GREAT JOB!


Step 3:
Config file sending
Currently send metrics in terminal, that is great! but we need to begin recording files for use in a demo.
This will include saving a json with scores + what the system passed + the actual image files.
These will be stored in a temp directory on the device (windows for test or linux for deploy)

For now, lets only focus on saving the json with score information.
If there are other preferred methods for sending info from luxonis pipeline to a file use that instead.

Once this is done we can discuss photo saving.

COMPLETED GREAT JOB

Step 4:
Add images to output associated with the json in logs 
Should save an image at a reasonable size, larger than inference, as a jpeg or png. 
When saved, we should be able to trace which image maps to which cell in the json.

Also, each run should make a test directory in whichever saving location

i.e we call the functions in logs/
it makes run_XXXXX with the json and image files inside 