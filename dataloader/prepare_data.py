import os
import numpy as np
import glob
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import transforms


def get_file_list(data_path):

    data_list = []

    if os.path.isfile(data_path):
        data_list.append(data_path)
    else:
        for subdir, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith((".tif")):
                    data_list.append(os.path.join(subdir, file))
    data_list.sort()
    if not data_list:
        raise FileNotFoundError("No data was found")
    return data_list


def balance_split(image_paths, classes, split_percentage):
    data = np.concatenate(
        [np.array(image_paths).reshape([-1, 1]), np.array(classes).reshape([-1, 1])],
        axis=1,
    )

    lst_paths_1 = []
    lst_paths_2 = []

    for class_ in np.unique(data[:, 1]):
        temp_data = data[data[:, 1] == class_]
        len_class = temp_data.shape[0]
        np.random.shuffle(temp_data)
        lst_paths_1.extend(temp_data[: round(len_class * split_percentage / 100)][:, 0])
        lst_paths_2.extend(temp_data[round(len_class * split_percentage / 100) :][:, 0])

        # print(f"len data1 : {len(temp_data[:round(len_class*split_percentage/100)])}")
        # print(f"len data2 : {len(temp_data[round(len_class*split_percentage/100):])}")

    return lst_paths_1, lst_paths_2


def train_test_val_split(root1, root2, split_percentage):
    """_summary_

    Args:
        root1 (str): based on CIFAR10 dataset, this is the root of initial train data
        root2 (str): based on CIFAR10 dataset, this is the root of initial test data
        split_percentage (float): Percentage to split root1 data,
                                  for example if 20%, first data will be 20% of root1 and the second one 80%
    """
    classes = os.listdir(root1)

    dirs1 = []
    dirs2 = []
    for _class in classes:
        dirs1 += glob.glob(root1 + _class + "/*.jpg")
        dirs2 += glob.glob(root2 + _class + "/*.jpg")

    # print("\nTotal train images: ", len(train_paths))
    # print("Total test images: ", len(test_paths))

    dirs1_1, dirs1_2 = balance_split(
        image_paths=dirs1,
        classes=[x.split("/")[-2] for x in dirs1],
        split_percentage=split_percentage,
    )

    return dirs1_1, dirs1_2, dirs2


class CIFAR10Dataset(Dataset):
    def __init__(self, imgs_paths, transforms=None):
        super(CIFAR10Dataset, self).__init__()
        classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        self.imgs_paths = imgs_paths
        self.class_to_int = {classes[i]: i for i in range(len(classes))}
        self.transforms = transforms

    def __getitem__(self, index):

        image_path = self.imgs_paths[index]

        # Reading image
        image = Image.open(image_path)

        # Retriving class label
        label = image_path.split("/")[-2]
        label = self.class_to_int[label]
        # Applying transforms on image
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label

    def __len__(self):
        return len(self.imgs_paths)

    def get_unique_labels(self):
        return list(self.class_to_int.values())

    def get_list_of_data(self):
        list_images = []
        list_labels = []

        loop_load = tqdm(
            enumerate(self.imgs_paths),
            total=len(self.imgs_paths),
            desc="Creating list of images and labels: ",
            position=0,
            leave=True,
        )

        for _, image_path in loop_load:
            image = Image.open(image_path)
            # Retriving class label
            label = image_path.split("/")[-2]
            label = self.class_to_int[label]
            # Applying transforms on image
            if self.transforms is not None:
                image = self.transforms(image)
            else:
                image = transforms.ToTensor()(image)

            list_images.append(image)
            list_labels.extend([label])
        return list_images, list_labels
