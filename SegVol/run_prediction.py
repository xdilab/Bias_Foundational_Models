import torch
from transformers import AutoModel, AutoTokenizer

# 1.) Setup: Load the Model and Set up the Device
print("Loading model... This might take a while.")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and its processor from the files I downloaded
# trust_remote_code=True is necessary to load the custom model code
clip_tokenizer = AutoTokenizer.from_pretrained("weights")
model = AutoModel.from_pretrained("weights", trust_remote_code=True, test_mode=True)
model.model.text_encoder.tokenizer = clip_tokenizer
model.eval() # Sets the pretrained model to evaluation mode.
model.to(device)
print("Model loaded successfully!")

# 2.) Configuration: Define the Paths & Prompts
# Define the path to the input image from my 'data' folder
ct_path = 'lgg_data/TCGA_CS_4943_0000.nii'
gt_path = 'lgg_data/TCGA_CS_4943.nii'

# Define the path for the output prediction file in my 'outputs' folder
save_path = 'outputs/prediction_lgg_patient_1.nii.gz'

# Define the text prompt. We're segmenting the liver.
# Define demo ground truth also has kidney, pancreas, and spleen information that I can test later.
categories = ["brain tumor", "lower grade glioma"]
print(f"Targeting segmentation for: {categories}")

# 3.) Preprocessing: Prepare the Data for the Model
print("Preprocessing the image...")
# The model's processor handles converting the .nii file into the format the model needs.
ct_npy, gt_npy = model.processor.preprocess_ct_gt(ct_path, gt_path, category=categories)
#ct_tensor = torch.tensor(ct_npy, device=device)

#go through zoom_transform to generate zoom-out and zoom-in views
data_item = model.processor.zoom_transform(ct_npy, gt_npy)

# add batch dim
data_item['image'], data_item['label'], data_item['zoom_out_image'], data_item['zoom_out_label'] = \
data_item['image'].unsqueeze(0).to(device), data_item['label'].unsqueeze(0).to(device), data_item['zoom_out_image'].unsqueeze(0).to(device), data_item['zoom_out_label'].unsqueeze(0).to(device)
print("Image preprocessed.")

# we're using one patient as an example.
cls_index = 0
text_prompt = [categories[cls_index]]

point_prompt, point_prompt_map = model.processor.point_prompt_b(data_item['zoom_out_label'][0][cls_index], device=device) # inputs w/o batch dim, outputs w batch dim
bbox_prompt, bbox_prompt_map = model.processor.bbox_prompt_b(data_item['zoom_out_label'][0][cls_index], device=device) # inputs w/o batch dim, outputs w batch dim

print("prompt done")

# Run the predictions.
print("Running prediction... This is the main step.")
 # The model's forward_test function takes the processed image and the text prompt
logits_mask = model.forward_test(image=data_item['image'], zoomed_image=data_item['zoom_out_image'], bbox_prompt_group=[bbox_prompt, bbox_prompt_map], text_prompt=text_prompt, use_zoom=True)

# calculate the dice score
dice = model.processor.dice_score(logits_mask[0][0], data_item['label'][0][cls_index], device)
print("Dice score is : " + str(dice))
print("Prediction complete!")

# 5.) Postprocessing: Save the Result
print(f"Saving the prediction mask to {save_path}")
# The model's processor has a handy function to save the output mask in the correct.nii format.
model.processor.save_preds(ct_path, save_path, logits_mask[0][0],  start_coord=data_item['foreground_start_coord'], end_coord=data_item['foreground_end_coord'])
print("\nDone! The results for 'prediction_liver.nii.gz' can be found in the 'outputs' folder.")