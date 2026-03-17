"""
Main script for closed-set device classification using CSI-based RFFI.

This script implements training and testing of a ResNet neural network for
device classification based on radio frequency fingerprints extracted from
5G PUSCH CSI. The method uses CSI obfuscation-inspired features to extract
device-specific fingerprints that are independent of the wireless channel.

Workflow:
1. Load CSI dataset and extract RFFI features
2. Train ResNet classification network
3. Evaluate on same-day and next-day test sets
4. Generate confusion matrices and accuracy metrics

This code is a fork of the LoRa RFFI repository:
https://github.com/gxhen/LoRa_RFFI/tree/main/Closed_set_RFFI

Core Modification: The main modification to the original repository is the implementation
of the Load5gDataset class in dataset_preparation.py, which loads 5G CSI data and
extracts CSI obfuscation-inspired RFFI features.

Note: You can modify the simulation code below to either train the NN with 5 or all 6 UEs.
Also, training is non-deterministic. Multiple training trials can achieve slightly different results.

@author: Based on work by G. Shen et al. (LoRa RFFI), adapted for 5G CSI and CSI obfuscation-inspired features
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop

from dataset_preparation import Load5gDataset

from deep_learning_models import classification_net
from keras.utils import to_categorical

import tensorflow as tf

# Set random seeds for reproducibility
tf.random.set_seed(1)
np.random.seed(1)

# GPU configuration
# TODO: Find out how to use multiple GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
GPU_NUM = 0  # Select which GPU to use
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = GPU_NUM
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)


def train(training_csi, training_labels, test_csi, test_labels, epochs=400):
    """
    Train a ResNet classification network for device classification.

    Parameters:
        training_csi: Training CSI features (RFFI features)
        training_labels: Training device class labels
        test_csi: Validation/test CSI features
        test_labels: Validation/test device class labels
        epochs: Maximum number of training epochs (default: 400)

    Returns:
        model: Trained classification neural network
    """

    # One-hot encoding
    label_train = training_labels
    label_one_hot = to_categorical(label_train)

    label_validate_one_hot = to_categorical(test_labels)

    data = training_csi

    # Learning rate scheduler
    early_stop = EarlyStopping('val_loss', min_delta=0, patience=30)
    reduce_lr = ReduceLROnPlateau('val_loss', min_delta=0, factor=0.2, patience=10, verbose=1)
    callbacks = [early_stop, reduce_lr]

    # Specify optimizer and deep learning model
    opt = RMSprop(learning_rate=1e-3)
    model = classification_net(data.shape, len(np.unique(label_train)))
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    # Start training
    history = model.fit(data,
                        label_one_hot,
                        epochs=epochs,
                        shuffle=True,
                        validation_data=(test_csi, label_validate_one_hot),
                        verbose=1,
                        batch_size=32,
                        callbacks=callbacks)

    return model


def test(clf_path_in, result_path, test_csi, test_labels, label_list, show_plots=True):
    """
    Test the trained classification network and generate evaluation metrics.

    Parameters:
        clf_path_in: Path to saved trained model
        result_path: Path prefix for saving results (confusion matrix plots and data)
        test_csi: Test CSI features
        test_labels: True device class labels for test set
        label_list: List of device class names for plotting

    Returns:
        acc: Overall classification accuracy
    """

    label_test = test_labels

    # Load neural network
    net_test = load_model(clf_path_in, compile=False, safe_mode=False)

    # Convert to channel independent spectrogram
    data = test_csi

    # Make prediction
    pred_prob = net_test.predict(data)
    pred_label = pred_prob.argmax(axis=-1)

    # Plot confusion matrix
    conf_mat = confusion_matrix(label_test, pred_label)
    classes = label_list

    plt.figure()
    sns.heatmap(conf_mat, annot=True,
                fmt='d', cmap='Blues',
                annot_kws={'size': 7},
                cbar=False,
                xticklabels=classes,
                yticklabels=classes)

    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    # result_path = "./results/confusion_matrix_no_random_subsampling10"
    plt.savefig(result_path+'.pdf', bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()

    rows, cols = conf_mat.shape
    with open(result_path + ".dat", "w") as f:
        # f.write("x y C\n")  # header
        for y in range(rows):
            for x in range(cols):
                f.write(f"{x} {y} {conf_mat[y, x]:.5f}\n")
    # np.savetxt(result_path + ".csv", conf_mat, delimiter=",")
    
    return accuracy_score(label_test, pred_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Closed-set device classification (5G CSI RFFI)")
    parser.add_argument("--experiment-name", default="5_ues_complex_features_val_middle_4oru_2026_03_17", type=str)
    parser.add_argument("--feature-type", default="obfuscation", choices=["obfuscation", "ofdm_absolutes"]) # Feature extraction type: 'obfuscation' or 'ofdm_absolutes'
    parser.add_argument("--file-path", default="/scratch/bsc26f18/datasets/device_classification_2025_11_15", type=str)
    parser.add_argument("--n-orus", default=4, type=int) # number of O-RUs (access points)
    parser.add_argument("--random-subsampling", action="store_true")
    parser.add_argument("--take-middle", action="store_true") # Take middle samples for validation set
    parser.add_argument("--test-to-all-ratio", default=0.125, type=float) # Fraction of samples used for same-day testing (12.5%)
    parser.add_argument("--epochs", default=400, type=int) # number of epochs to train for

    # Data subsetting for smoke tests / quick iterations
    parser.add_argument("--subset-fraction", default=None, type=float)
    parser.add_argument("--max-samples-per-label", default=None, type=int)
    parser.add_argument("--subset-seed", default=1, type=int)

    # UX
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib windows")
    parser.add_argument("--smoke", action="store_true", help="Tiny subset + very few epochs (fast pipeline check)")
    args = parser.parse_args()

    if args.smoke:
        # Keep it tiny and fast; goal is end-to-end correctness, not accuracy.
        if args.subset_fraction is None:
            args.subset_fraction = 0.02
        if args.max_samples_per_label is None:
            args.max_samples_per_label = 20
        args.epochs = min(args.epochs, 1)
        args.random_subsampling = True
        args.take_middle = False
        args.test_to_all_ratio = 0.2
        args.no_show = True
        args.experiment_name = args.experiment_name + "_SMOKE"

    # Configuration parameters
    experiment_name = args.experiment_name
    feature_type = args.feature_type
    random_subsampling = args.random_subsampling
    take_middle = args.take_middle
    test_to_all_ratio = args.test_to_all_ratio
    file_path = args.file_path
    n_orus = args.n_orus

    # uncomment this to train on all 6 UEs
    # label_list_training = ["iPhone14Pro_gold", "iPhone14Pro_black", "iPhone16e", "OnePlusNord", "sgs23", "pixel7"]
    # train only on 5 UEs
    label_list_training = ["iPhone14Pro_gold", "iPhone16e", "OnePlusNord", "sgs23", "pixel7"]
    
    # Keras 3 requires a recognized extension for model.save()
    clf_path = f'/scratch/bsc26f18/results/weights/cnn_subsampling_{random_subsampling}_features_{feature_type}_{experiment_name}.keras'
    results_path = f'/scratch/bsc26f18/results/device_classification/confusion_matrix_{random_subsampling}_features_{feature_type}_{experiment_name}'

    # Load dataset and extract features
    LoadDatasetObj = Load5gDataset(feature_type=feature_type)
    training_csi, training_labels, test_csi, test_labels = \
         LoadDatasetObj.load_channel_estimates(file_path=file_path+"/training", 
                                               label_list=label_list_training,
                                               random_subsampling=random_subsampling,
                                               n_orus=n_orus,
                                               test_to_all_ratio=test_to_all_ratio,
                                               take_middle=take_middle,
                                               subset_fraction=args.subset_fraction,
                                               max_samples_per_label=args.max_samples_per_label,
                                               subset_seed=args.subset_seed)
    
    # Train classification network
    clf_net = train(training_csi, training_labels, test_csi, test_labels, epochs=args.epochs)
    clf_net.save(clf_path)

    # if random_subsampling==False:
    #     label_list_testing = ["iphone14pro_gold", "iphone16e", "oneplusNord_eth", "sgs23", "googlePixel7"] # ["iphone14pro_gold", "iphone15pro", "iphone16e", "oneplusNord_eth", "sgs23", "googlePixel7", "iphone14pro_black",  "iPhone15Pro_2nd_measurement", "iPhone15Pro_3rd_measurement_monday", "sgs23_monday_uestand"]
    #     training_csi, training_labels, test_csi, test_labels = \
    #         LoadDatasetObj.load_channel_estimates(file_path=file_path, 
    #                                             label_list=label_list_testing,
    #                                             random_subsampling=False,
    #                                             n_orus=n_orus,
    #                                             test_to_all_ratio=0.25)
    
    # Evaluate on same-day test set
    label_list_testing_plot = label_list_training
    acc = test(clf_path, results_path, test_csi, test_labels, label_list_testing_plot, show_plots=not args.no_show)
    print('Overall accuracy (same day, 5) = %.4f' % acc)

    # Load next-day test set and evaluate
    _, _, test_csi, test_labels = \
         LoadDatasetObj.load_channel_estimates(file_path=file_path+"/testing", 
                                               label_list=label_list_training,
                                               random_subsampling=random_subsampling,
                                               n_orus=n_orus,
                                               test_to_all_ratio=1.0,
                                               take_middle=take_middle,
                                               subset_fraction=args.subset_fraction,
                                               max_samples_per_label=args.max_samples_per_label,
                                               subset_seed=args.subset_seed)
    
    acc = test(clf_path, results_path+"_next_day", test_csi, test_labels, label_list_testing_plot, show_plots=not args.no_show)
    print('Overall accuracy (next day, 5) = %.4f' % acc)

    # label_list_training = ["iPhone14Pro_gold", "iPhone16e", "OnePlusNord", "sgs23", "pixel7"] # "iPhone14Pro_black"
    label_list_testing6 = ["iPhone14Pro_gold", "iPhone16e", "OnePlusNord", "sgs23", "pixel7", "iPhone14Pro_black"]
    _, _, test_csi, test_labels = \
         LoadDatasetObj.load_channel_estimates(file_path=file_path+"/testing", 
                                               label_list=label_list_testing6,
                                               random_subsampling=random_subsampling,
                                               n_orus=n_orus,
                                               test_to_all_ratio=1.0,
                                               take_middle=take_middle,
                                               subset_fraction=args.subset_fraction,
                                               max_samples_per_label=args.max_samples_per_label,
                                               subset_seed=args.subset_seed)
    
    acc = test(clf_path, results_path+"_next_day_6", test_csi, test_labels, label_list_testing6, show_plots=not args.no_show)
    print('Overall accuracy (next day, 6) = %.4f' % acc)

    # label_list_training = ["iPhone14Pro_gold", "iPhone16e", "OnePlusNord", "sgs23", "pixel7"] # "iPhone14Pro_black"
    # label_list_testing5 = ["iPhone14Pro_gold", "iPhone16e", "OnePlusNord", "sgs23", "pixel7", "iPhone14Pro_black"]
    # _, _, test_csi, test_labels = \
    #      LoadDatasetObj.load_channel_estimates(file_path=file_path+"/training", 
    #                                            label_list=label_list_testing5,
    #                                            random_subsampling=random_subsampling,
    #                                            n_orus=n_orus,
    #                                            test_to_all_ratio=test_to_all_ratio,
    #                                            take_middle=take_middle)
    
    # acc = test(clf_path, results_path+"_same_day_6", test_csi, test_labels, label_list_testing5)
    # print('Overall accuracy (same day, 6) = %.4f' % acc)
