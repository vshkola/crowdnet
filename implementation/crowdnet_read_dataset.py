from __future__ import print_function

from google.cloud import storage
from tensorflow.python.platform import gfile
from sklearn.model_selection import train_test_split
import pandas as pd

import argparse, logging
import os

gclient = storage.Client(project='lv-images')

ACCEPTED_USER_BLOB_PREFIX = 'users/pictures/'
DECLINED_USER_BLOB_PREDIX = 'deleted/'
BUCKET_NAME = 'img.lovoo.com'
bucket = None

logging.basicConfig()
LOG = logging.getLogger("crowdnet-read-dataset")
LOG.setLevel(logging.INFO)

class DataSet:
    def __init__(self, dataset, number_of_samples=None):
        if number_of_samples == None:
            self.dataset = dataset
        elif number_of_samples <= dataset.size:
            self.number_of_samples = number_of_samples
            self.dataset = dataset[0:number_of_samples]
        else:
            LOG.error('Error: dataset is smaller than required amount of samples')


    def _get_dataset_ids(self, samples):
            images       = samples['image_id'].tolist()
            labels       = samples['label'].tolist()
            labels_group = samples['label_group'].tolist()
            return  {'images': images,
                    'labels': labels,
                    'labels_group': labels_group}


    def _split_into_train_test_validation(self):
        train, raw_test = train_test_split(self.dataset, test_size=0.45, stratify=self.dataset['label_group'])
        test, validation = train_test_split(raw_test, test_size=0.2, stratify=raw_test['label_group'])
        return [train, test, validation]


    def _download_images(self, dir_name, samples):
        dir_to_save = os.path.join(self.target_dir, dir_name)

        if gfile.Exists(dir_to_save):
            LOG.info("{} already exist".format(dir_name))
        else:
            LOG.info("{} not exist, will create {} samples in dataset".format(dir_name, len(samples)))

            dataset_ids = self._get_dataset_ids(samples)
            images, labels, groups = dataset_ids.items()

            for image_id, group in zip(images[1], groups[1]):
                image_blob = get_img_blob(group == 1, image_id)
                if image_blob is not None and image_blob.exists() is True:
                    save_blob_to_dir(image_blob, image_id, dir_to_save, group)
                    LOG.info('{}: {}, {} exist'.format(dir_name, image_id, group))
                else:
                    LOG.info('{}: {}, {} not exist'.format(dir_name, image_id, group))




    def maybe_download_images(self, target_dir):
        self.target_dir = target_dir
        #splitted_samples = self._split_into_train_test_validation()

        #self._download_images('train',      splitted_samples[0])
        #self._download_images('test',       splitted_samples[1])
        #self._download_images('validation', splitted_samples[2])
        self._download_images('validation', self.dataset)


    def save_dataset_labels(self):
        target_path = os.path.join(self.target_dir, "labels_{}.csv".format(self.number_of_samples))
        col_list = ['label','label_group','image_id']
        self.dataset[0:self.number_of_samples][col_list].to_csv(target_path, quoting=None, index=True)

def init_img_bucket(bucket_name):
    global bucket
    bucket = gclient.bucket(bucket_name)


def get_blob_name(is_accepted, img_id):
    if is_accepted:
        return '{}{}/image.jpg'.format(ACCEPTED_USER_BLOB_PREFIX, img_id)
    else:
        return '{}{}/image.jpg'.format(DECLINED_USER_BLOB_PREDIX, img_id)


def get_img_blob(is_accepted, img_id):
    if bucket == None:
        LOG.error("Error: Bucket not initialzed")
    blob_name = get_blob_name(is_accepted, img_id)
    return bucket.get_blob(blob_name)


def save_blob_to_dir(img_blob, file_name, target_dir, group):
    # TODO Check image size and if < 5Kb - remove from dataset. Especially valid for negative cases
    if group == 1:
        label_name = "accepted"
    else:
        label_name = "declined"

    label_dir_path = os.path.join(target_dir, label_name)
    if not gfile.Exists(label_dir_path):
        gfile.MakeDirs(label_dir_path)

    image_path = '{}/{}.jpg'.format(label_dir_path, file_name)

    if gfile.Exists(image_path):
        LOG.info("Image {} already exist in destination".format(image_path))
    else:
        with open(image_path, 'wb') as file_obj:
            img_blob.download_to_file(file_obj)


def load_csv_dataset(csv_dataset_path):
    return pd.DataFrame.from_csv(csv_dataset_path)


def read_dataset():
    parser = argparse.ArgumentParser(description="image downloader")

    parser.add_argument('--size',
                        dest='dataset_size',
                        type = int,
                        default=None,
                        help="number of samples in dataset")
    parser.add_argument('--ids',
                        dest='dataset_path',
                        type = str,
                        default='dataset_middle_validation.csv',
                        help="path to the csv file with image to label mapping")
    parser.add_argument('--target_dir',
                        dest='target_dir',
                        type = str,
                        default='images_big',
                        help="path to target dir where to save dataset images")
    parser.add_argument('--test_size',
                        dest='test_size',
                        type=int,
                        default=25,
                        help="percentage of required test images")
    parser.add_argument('--validation_size',
                        dest='validation_size',
                        type=int,
                        default=15,
                        help="percentage of required validation images")

    FLAGS, unknown = parser.parse_known_args()

    init_img_bucket(BUCKET_NAME)
    csv_dataset = load_csv_dataset(FLAGS.dataset_path)

    dataset = DataSet(csv_dataset, FLAGS.dataset_size)
    dataset.maybe_download_images(FLAGS.target_dir)
    #dataset.save_dataset_labels()

if __name__ == "__main__":
    read_dataset()

