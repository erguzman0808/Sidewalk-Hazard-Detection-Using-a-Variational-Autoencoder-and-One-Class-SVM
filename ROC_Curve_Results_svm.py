import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from VAE import CNN_VAE  # Ensure this is the correct import
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import pandas as pd
from itertools import product  # For iterating over threshold pairs
from scipy.spatial import ConvexHull

# Function to load and preprocess an image
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4500, 0.4552, 0.4417], std=[0.0930, 0.0883, 0.0936])
    ])
    img = Image.open(image_path).convert('RGB')  # Convert to RGB
    return transform(img)  # Apply the transformations

def tensor_to_numpy_processing(tensor):
    # Reverse the normalization for visualization
    mean = np.array([0.4500, 0.4552, 0.4417])
    std = np.array([0.0930, 0.0883, 0.0936])
    tensor = tensor.detach().cpu().numpy().squeeze()
    tensor = (tensor * std[:, None, None]) + mean[:, None, None]  # Broadcasting for CHW format
    return tensor

# Convert a tensor back to a numpy array for visualization
def tensor_to_numpy(tensor):
    mean = np.array([0.4500, 0.4552, 0.4417])
    std = np.array([0.0930, 0.0883, 0.0936])
    tensor = tensor.detach().cpu().numpy().squeeze()
    tensor = (tensor * std[:, None, None]) + mean[:, None, None]  # Broadcasting for CHW format
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)  # Convert to [0, 255] range and to uint8
    if tensor.shape[0] == 3:  # Convert CHW to HWC
        tensor = np.transpose(tensor, (1, 2, 0))
    return tensor

# Loss function for VAE: combines reconstruction loss and KL divergence
def loss_func(recon_x, x, mu, logvar):
    mse = np.sum((recon_x - x) ** 2)
    kld = -0.5 * np.sum(1 + logvar - np.square(mu) - np.exp(logvar))
    return mse + kld, mse, kld  # Total VAE loss

def compute_auc_upper_hull(points):
    sorted_points = points[np.argsort(points[:, 0])]
    # Compute AUC using the trapezoidal rule
    area = np.trapz(sorted_points[:, 1], sorted_points[:, 0])
    return area

def get_outermost_points_hull(x, y):
    points = np.column_stack((x, y))  # Combine x and y into a single array
    hull = ConvexHull(points)  # Compute the convex hull
    # Extract the outermost points using the hull indices
    outer_points = points[hull.vertices]
    return outer_points

def get_outermost_points(fpr, tpr, add_endpoints=True):
    # Combine into dataframe
    df = pd.DataFrame({"FPR": fpr, "TPR": tpr})

    # Clean numeric and clip range
    df = df.dropna()
    df["FPR"] = pd.to_numeric(df["FPR"], errors="coerce").clip(0, 1)
    df["TPR"] = pd.to_numeric(df["TPR"], errors="coerce").clip(0, 1)

    # Get max TPR per unique FPR
    upper = df.groupby("FPR", as_index=False)["TPR"].max().sort_values("FPR")

    # Convert to NumPy array
    upper_points = upper.to_numpy()
    upper_points = get_outermost_points_hull(upper_points[:,0], upper_points[:,1])

    return upper_points

# Plot for verification
def plot_upper_envelope(fpr, tpr):
    upper_points = get_upper_envelope(fpr, tpr)
    plt.figure(figsize=(7.5, 6))
    plt.scatter(fpr, tpr, s=10, alpha=0.3, label="All points", color="red")
    plt.plot(upper_points[:, 0], upper_points[:, 1], color="orange", linewidth=2.5, label="Upper envelope")
    plt.scatter(upper_points[:, 0], upper_points[:, 1], color="orange", s=25)


def save_image(numpy_img, path):
    """Saves a NumPy array as an image."""
    image = Image.fromarray(numpy_img)
    image.save(path)

# Function to process images and get TPR and FPR for each threshold
def process_images_and_get_metrics(image_folder, model, device, KLD_thre, MSE_thre, ocsvm, scalar, ocsvm_only, scalar_only, ground_truth_list):
    tp = 'tp'
    tn = 'tn'
    fp = 'fp'
    fn = 'fn'
    true_labels = []
    #predicted_scores = []
    mse_scores = []
    decision_scores = []
    decision_scores_only = []

    # Ensure directories exist
    os.makedirs(tp, exist_ok=True)
    os.makedirs(tn, exist_ok=True)
    os.makedirs(fp, exist_ok=True)
    os.makedirs(fn, exist_ok=True)

    i = 0  # Start index
    for image_name in sorted(os.listdir(image_folder)):
        if 'leaves' in image_name or 'construction' in image_name:
            continue
        print(f"Index: {i}, Image name: {image_name}")
        image_path = os.path.join(image_folder, image_name)
        image_tensor = load_image(image_path).unsqueeze(0)
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            recon_img, mu, logvar, latent_vector, _ = model(image_tensor)            
            original = tensor_to_numpy_processing(image_tensor.squeeze(0))
            reconstructed = tensor_to_numpy_processing(recon_img.squeeze(0))

            mu = mu.detach().cpu().numpy()
            logvar = logvar.detach().cpu().numpy()            
            loss, mse, kld = loss_func(reconstructed, original, mu, logvar)
            test_data = scalar.transform(latent_vector.detach().cpu().numpy())
            decision = ocsvm.decision_function(test_data).item()

            test_data_only = scalar_only.transform(latent_vector.detach().cpu().numpy())
            decision_only = ocsvm_only.decision_function(test_data_only).item()

            mse_scores.append(mse)
            decision_scores.append(decision)
            decision_scores_only.append(decision_only)

            
            # Check for anomaly
            if mse  > MSE_thre:  # kld > KLD_thre or mse > MSE_thre
                anomaly_score = 1
                print('Anomaly')
            else:
                anomaly_score = 0  # If KLD is not significant, consider it as non-anomalous
                print('Not anomaly')
                # Save the image to the non-anomaly folder
                #save_image(original, os.path.join(non_anomaly_folder, image_name))
            
            #predicted_scores.append(anomaly_score)
            true_labels.append(ground_truth_list[i])
            i += 1
    #print(predicted_scores)
    #print(f'Threshold: {MSE_thre}')
    print(f'Total images processed: {i + 1}')
    #print(reconstruction_probability_avg/i+1)
    return np.array(true_labels), np.array(mse_scores), np.array(decision_scores), np.array(decision_scores_only)



# Function to generate and plot the ROC curve correctly
def generate_and_plot_roc_correct(
    image_folder, model, device,
    kld_threshold, mse_thresholds,    # kept for API compatibility (unused here)
    ocsvm, scalar,                    # OCSVM_aux (+ scaler) used with VAE
    ocsvm_only, scalar_only,          # OCSVM_solo (+ scaler) standalone
    ground_truth_list
):
    # Expect process_images_and_get_metrics to return 4 arrays:
    # y_true, mse_scores, decision_scores_aux, decision_scores_solo
    true_labels, predicted_scores, decision_scores, decision_scores_only = process_images_and_get_metrics(
        image_folder, model, device, kld_threshold, mse_thresholds,
        ocsvm, scalar, ocsvm_only, scalar_only, ground_truth_list
    )

    # Build threshold grids
    min_score1, max_score1 = float(np.min(predicted_scores)), float(np.max(predicted_scores))
    min_score2, max_score2 = float(np.min(decision_scores)),  float(np.max(decision_scores))
    min_score3, max_score3 = float(np.min(decision_scores_only)), float(np.max(decision_scores_only))

    print(f'Scores for VAE: {min_score1,max_score1}')
    print(f'Scores for SVM_only: {min_score3,max_score3}')

    thresholds1 = np.linspace(min_score1, max_score1, 500)  # VAE (MSE)
    thresholds2 = np.linspace(min_score2, max_score2, 500)  # OCSVM_aux (for fusion with VAE)
    thresholds3 = np.linspace(min_score3, max_score3, 500)  # OCSVM_solo (standalone)

    # Storage — separate per method
    # VAE-only
    fpr_list, tpr_list, precision_list, recall_list, f1_list = [], [], [], [], []
    threshold_mse_list = []

    # OCSVM_solo-only
    fpr_list_solo, tpr_list_solo, precision_list_solo, recall_list_solo, f1_list_solo = [], [], [], [], []
    threshold_solo_list = []

    # VAE + OCSVM_aux
    fpr_list_fuse, tpr_list_fuse, precision_list_fuse, recall_list_fuse, f1_list_fuse = [], [], [], [], []
    threshold_mse_fuse, threshold_aux_fuse = [], []

    best_f1, best_mse = 0.0, None
    best_f1_solo, best_solo = 0.0, None
    best_f1_fuse, best_mse_fuse, best_aux_fuse = 0.0, None, None

    # --- VAE-only sweep (thresholds1) ---
    for th_mse in thresholds1:
        preds = (predicted_scores > th_mse).astype(int)
        tn, fp, fn, tp = confusion_matrix(true_labels, preds, labels=[0, 1]).ravel()
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        P = tp / (tp + fp) if (tp + fp) > 0 else 0
        R = TPR
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0

        fpr_list.append(FPR); tpr_list.append(TPR)
        precision_list.append(P); recall_list.append(R); f1_list.append(F1)
        threshold_mse_list.append(th_mse)

        if F1 > best_f1:
            best_f1, best_mse = F1, th_mse

    # --- OCSVM_solo-only sweep (thresholds3) ---
    for th_solo in thresholds3:
        # For OneClassSVM, more negative decision_function => more anomalous
        preds_solo = (decision_scores_only < th_solo).astype(int)
        tn, fp, fn, tp = confusion_matrix(true_labels, preds_solo, labels=[0, 1]).ravel()
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        P = tp / (tp + fp) if (tp + fp) > 0 else 0
        R = TPR
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0

        fpr_list_solo.append(FPR); tpr_list_solo.append(TPR)
        precision_list_solo.append(P); recall_list_solo.append(R); f1_list_solo.append(F1)
        threshold_solo_list.append(th_solo)

        if F1 > best_f1_solo:
            best_f1_solo, best_solo = F1, th_solo

    # --- VAE + OCSVM_aux 2D sweep (thresholds1 x thresholds2) ---       
    for th_mse in thresholds1:
        for th_aux in thresholds2:
            #preds_fuse = ((predicted_scores > th_mse) & (decision_scores < th_aux)).astype(int)#(decision_scores < th_aux).astype(int) #
            vae_pos = predicted_scores > th_mse          # VAE flags anomaly
            preds_fuse = np.zeros_like(predicted_scores, dtype=int)

            # Only consult OCSVM on the VAE-positive subset
            if np.any(vae_pos):
                preds_fuse[vae_pos] = (decision_scores[vae_pos] < th_aux).astype(int)

            # preds_fuse is 1 only if BOTH: VAE positive AND OCSVM confirms
            # tn, fp, fn, tp = confusion_matrix(true_labels, preds_fuse, labels=[0, 1]).ravel()

            tn, fp, fn, tp = confusion_matrix(true_labels, preds_fuse, labels=[0, 1]).ravel()
            TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
            FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
            P = tp / (tp + fp) if (tp + fp) > 0 else 0
            R = TPR
            F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0

            fpr_list_fuse.append(FPR); tpr_list_fuse.append(TPR)
            precision_list_fuse.append(P); recall_list_fuse.append(R); f1_list_fuse.append(F1)
            threshold_mse_fuse.append(th_mse); threshold_aux_fuse.append(th_aux)

            if F1 > best_f1_fuse:
                best_f1_fuse, best_mse_fuse, best_aux_fuse = F1, th_mse, th_aux

    # --- Build upper envelopes and AUCs ---
    points_roc_vae = get_outermost_points(fpr_list, tpr_list)
    points_pr_vae  = get_outermost_points(recall_list, precision_list)
    roc_auc = compute_auc_upper_hull(points_roc_vae)
    pr_auc  = compute_auc_upper_hull(points_pr_vae)

    points_roc_solo = get_outermost_points(fpr_list_solo, tpr_list_solo)
    points_pr_solo  = get_outermost_points(recall_list_solo, precision_list_solo)
    roc_auc_solo = compute_auc_upper_hull(points_roc_solo)
    pr_auc_solo  = compute_auc_upper_hull(points_pr_solo)

    points_roc_fuse = get_outermost_points(fpr_list_fuse, tpr_list_fuse)
    points_pr_fuse  = get_outermost_points(recall_list_fuse, precision_list_fuse)
    roc_auc_fuse = compute_auc_upper_hull(points_roc_fuse)
    pr_auc_fuse  = compute_auc_upper_hull(points_pr_fuse)

    # --- Prints (best thresholds) ---
    print(f"Best F1 VAE-only: {best_f1:.3f} at MSE_thr={best_mse:.6f}")
    print(f"Best F1 OCSVM_solo: {best_f1_solo:.3f} at SOLO_thr={best_solo:.6f}")
    print(f"Best F1 VAE+OCSVM_aux: {best_f1_fuse:.3f} at MSE_thr={best_mse_fuse:.6f}, AUX_thr={best_aux_fuse:.6f}")

    # --- Plots ---
    plt.figure(figsize=(12, 6))
    plt.scatter(points_roc_solo[:, 0], points_roc_solo[:, 1], marker='o', color='darkgreen', lw=3,
                label=f'ROC OCSVM_solo (AUC = {roc_auc_solo:.3f})')
    plt.scatter(points_roc_vae[:, 0], points_roc_vae[:, 1], marker='o', color='darkblue', lw=2,
                label=f'ROC VAE (AUC = {roc_auc:.3f})')
    plt.scatter(points_roc_fuse[:, 0], points_roc_fuse[:, 1], marker='o', color='darkorange', lw=3,
                label=f'ROC VAE+OCSVM (AUC = {roc_auc_fuse:.3f})')
    #plt.scatter(fpr_list_fuse, tpr_list_fuse, marker='o', color='darkorange', lw=3,
    #            label=f'ROC VAE+OCSVM (AUC = {roc_auc_fuse:.3f})')   
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("FPR (False Positive Rate)"); plt.ylabel("TPR (True Positive Rate)")
    plt.title('Depth Receiver Operating Characteristic')
    plt.legend(loc="lower right"); plt.grid(True); plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=300); plt.show()

    plt.figure(figsize=(12, 6))
    plt.scatter(points_pr_solo[:, 0], points_pr_solo[:, 1], marker='o', color='darkgreen', lw=2,
                label=f'PR OCSVM_solo (AUC = {pr_auc_solo:.3f})')
    plt.scatter(points_pr_vae[:, 0], points_pr_vae[:, 1], marker='o', color='darkblue', lw=2,
                label=f'PR VAE (AUC = {pr_auc:.3f})')
    plt.scatter(points_pr_fuse[:, 0], points_pr_fuse[:, 1], marker='o', color='darkorange', lw=2,
                label=f'PR VAE+OCSVM (AUC = {pr_auc_fuse:.3f})')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title('Depth Precision v Recall Curve')
    plt.legend(loc="lower right"); plt.grid(True); plt.tight_layout()
    plt.savefig("precision_recall_curve.png", dpi=300); plt.show()

    # --- Save sweep results to SEPARATE CSVs ---
    # 1) VAE-only (M entries)
    df_vae = pd.DataFrame({
        'MSE_thr': threshold_mse_list,
        'TPR': tpr_list,
        'FPR': fpr_list,
        'Precision': precision_list,
        'Recall': recall_list,
        # 'F1_vae': f1_list,  # uncomment if you tracked it
    })
    df_vae.to_csv('metrics_vae_brick.csv', index=False)

    # 2) OCSVM_solo-only (K entries)
    df_solo = pd.DataFrame({
        'SOLO_thr': threshold_solo_list,
        'TPR_svm': tpr_list_solo,
        'FPR_svm': fpr_list_solo,
        'Precision_svm': precision_list_solo,
        'Recall_svm': recall_list_solo,
        # 'F1_svm': f1_list_solo,
    })
    df_solo.to_csv('metrics_ocsvm_solo_brick.csv', index=False)

    # 3) VAE + OCSVM_aux (M×K entries)
    df_fuse = pd.DataFrame({
        'MSE_thr_fuse': threshold_mse_fuse,
        'AUX_thr': threshold_aux_fuse,
        'TPR_vae+svm': tpr_list_fuse,
        'FPR_vae+svm': fpr_list_fuse,
        'Precision_vae+svm': precision_list_fuse,
        'Recall_vae+svm': recall_list_fuse,
        # 'F1_vae+svm': f1_list_fuse,
    })
    df_fuse.to_csv('metrics_fuse_brick.csv', index=False)

    print("Saved separate metric files:")
    print(" - metrics_vae.csv")
    print(" - metrics_ocsvm_solo.csv")
    print(" - metrics_fuse.csv")


# Main function
def main():
    # Load your trained model, ocsvm, and scaler
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')# torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the model and load weights
    model = CNN_VAE()  # Assuming your VAE model class is CNN_VAE
    model.load_state_dict(torch.load('vae_examples/vae_out_all_data.pth', map_location=device))#vae_examples/vae_cover_not_aug.pth
    model.to(device)
    model.eval()


    ocsvm = joblib.load("ocsvm_model.pkl")
    scalar = joblib.load("scaler.pkl")
    ocsvm_only = joblib.load("ocsvm_model_brick_cement.pkl")
    scalar_only = joblib.load("scaler_brick_cement.pkl")

    # Define the image folder
    #image_folder = '/Users/edgarguzman/Documents/PHD/Anomaly/Code/Data_Collection/Models/Data/output_images/Outdoor_Validation/color2'
    image_folder = '/Users/edgarguzman/Documents/PHD/Anomaly/Code/Data_Collection/Models/Data/output_images/Outdoor_Validation/color_brick'
    #image_folder = '/Users/edgarguzman/Documents/PHD/Anomaly/Code/Data_Collection/Models/Data/output_images/brickano/color'

    # Provide ground truth labels (1 for anomaly, 0 for normal)
    '''
    This if for Cement
    '''
    bottle_ground_truth_list = np.zeros(96)
    bottle_ground_truth_list[:15] = 1
    bottle_ground_truth_list[44:] = 1

    branch_ground_truth_list = np.zeros(83)
    branch_ground_truth_list[30:] = 1

    gravel_ground_truth_list = np.ones(269)
    gravel_ground_truth_list[137:165] = 0

    manhole = np.zeros(419)

    validation_ground_truth_list = np.zeros(2514)
    validation_ground_truth_list[200:219] = 1
    validation_ground_truth_list[872:1051] = 1
    validation_ground_truth_list[2130:2197] = 1
    validation_ground_truth_list[2209:2222] = 1
    validation_ground_truth_list[2232:2239] = 1
    validation_ground_truth_list[2280:2317] = 1

    pothole_ground_truth_list = np.zeros(468)
    pothole_ground_truth_list[134:174] = 1
    pothole_ground_truth_list[208:271] = 1
    pothole_ground_truth_list[290:392] = 1
    pothole_ground_truth_list[400:] = 1

    uneven_ground_truth_list = np.zeros(396)
    uneven_ground_truth_list[80:220] = 1
    uneven_ground_truth_list[322:] = 1

    Lemon = np.ones(176)
    
    mexico_gravel = np.zeros(331)
    mexico_gravel[190:] = 1

    dirt = np.zeros(1005)
    dirt[110:267] = 1
    dirt[729:855] = 1

    snow = np.zeros(245)
    snow[86:] = 1

    # Combine all ground truth lists into a single array
    ground_truth_list = np.concatenate([
        bottle_ground_truth_list,
        branch_ground_truth_list,
        dirt,
        gravel_ground_truth_list,
        Lemon,
        manhole,
        mexico_gravel,
        pothole_ground_truth_list,
        snow,
        uneven_ground_truth_list,
        validation_ground_truth_list
    ])
    

    '''
    This is for Brick
    '''

    
    anomaly = np.ones(639)
    ice = np.zeros(150)
    
    brick_ano = np.zeros(1298)
    brick_ano[100:312] = 1
    brick_ano[330:403] = 1
    brick_ano[430:450] = 1
    brick_ano[579:648] = 1
    brick_ano[773:850] = 1
    brick_ano[890:970] = 1
    brick_ano[1074:1146] = 1
    brick_ano[1260:] = 1
    # Combine all ground truth lists into a single array
    ground_truth_list = np.concatenate([
        anomaly,
        brick_ano,
        ice,

    ])
    
    
    print(ground_truth_list.shape)
    # Define the MSE threshold and a range of KLD thresholds to evaluate
    kld_threshold = 180  # You can choose a fixed MSE threshold
    mse_thresholds = [529.739782]#np.linspace(0, 350, 5)  # More fine-grained KLD thresholds

    # Generate and plot the ROC curve correctly
    #generate_and_plot_roc_correct(image_folder, model, device, kld_threshold, mse_thresholds, ocsvm, 
    #                              scalar, ground_truth_list)

    generate_and_plot_roc_correct(image_folder, model, device, kld_threshold, mse_thresholds, ocsvm, 
                                  scalar, ocsvm_only, scalar_only, ground_truth_list)
if __name__ == "__main__":
    main()
