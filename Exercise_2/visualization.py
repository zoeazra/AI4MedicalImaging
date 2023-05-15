import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def visualize_input_and_prediction(input_path, prediction_path):
    input_array = nib.load(input_path).get_fdata()
    input_array = 255 * (input_array - np.min(input_array)) / np.max(input_array)
    prediction_array = nib.load(prediction_path).get_fdata()
    prediction_array = 255 * (prediction_array - np.min(prediction_array)) / np.max(prediction_array)

    # Create two subplots and unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(input_array.squeeze())
    ax2.imshow(prediction_array.squeeze())
    plt.show()

def show_data(dataset,index,n_images_display = 5, message = 'Raw data'):
    fig, ax = plt.subplots(1, n_images_display, figsize=(20, 5))
    for i in range(n_images_display):
        image, label = dataset[index + i]
        ax[i].imshow(np.uint8(image))
        # ax[i].imshow(np.uint8(np.transpose(image, (1, 2, 0))));
        ax[i].set_title(f'Cancer : {"Yes" if label else "No"}')
        ax[i].axis('off')
    plt.suptitle(
        f'{message} of Skin Cancer Images [Indices: {index} - {index + n_images_display} - Images Shape: {dataset[0][0].shape}]');
    plt.show()


def show_data_Unet(dataset,index,n_images_display = 5):
    # @title Visualizing Images { run: 'auto'}
    fig, ax = plt.subplots(1, n_images_display, figsize=(20, 5))
    for i in range(n_images_display):
        sample = dataset[index + i]
        image, mask = sample['image'], sample['mask']
        image, mask = np.uint8(image), np.uint8(mask)
        edge = cv2.Canny(mask * 255, 100, 200)
        image[edge > 0] = (0, 255, 0)
        ax[i].imshow(image)
        ax[i].axis('off')
    plt.suptitle(
        f"Skin Cancer Images [Indices: {index} - {index + n_images_display} - Images Shape: {dataset[0]['image'].shape}]");
    plt.show()

def show_augmentation_data(data_module,index=0,n_images_display = 5):
    # initialize dataloaders
    data_module.setup()
    traindata = data_module.train_dataloader().dataset
    # dimensions need to be tranposed
    traindata_T = []
    i=0
    for image, label in traindata:
        if i>index-1:
            if i<n_images_display+index:
                traindata_T.append((np.array(image).transpose(1, 2, 0), np.array(label)))
    show_data(traindata_T,index=0,n_images_display=n_images_display,message = 'Augmented data')
    return
