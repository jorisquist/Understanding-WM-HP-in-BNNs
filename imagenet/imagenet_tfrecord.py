import logging
import os.path
from subprocess import call

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali as dali
    import nvidia.dali.fn as fn
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    import nvidia.dali.tfrecord as tfrec
except ImportError:
    logging.exception("Please install DALI from https://www.github.com/NVIDIA/DALI to use this dataloader.")


def ImageNet_TFRecord_loader(root, split, batch_size, num_threads, device_id, num_gpus,
                      dali_cpu=False, augment=False):
    """
    PyTorch dataloader for ImageNet TFRecord files.

    Args:
        root (str): Location of the 'tfrecords' ImageNet directory.
        split (str): Split to use, either 'train' or 'val'.
        batch_size (int): Batch size per GPU (default=64).
        num_threads (int): Number of dataloader workers to use per sub-process.
        device_id (int): ID of the GPU corresponding to the current subprocess. Dataset
            will be divided over all subprocesses.
        num_gpus (int): Total number of GPUS available.
        dali_cpu (bool): Set True to perform part of data loading on CPU instead of GPU (default=False).
        augment (bool): Whether or not to apply data augmentation (random cropping,
            horizontal flips).

    Returns:
        PyTorch dataloader.

    """
    # List all tfrecord files in directory
    tf_files = os.listdir(os.path.join(root, split, 'data'))

    # Create dir for idx files if not exists
    idx_files_dir = os.path.join(root, split, 'idx_files')
    if not os.path.exists(idx_files_dir):
        os.mkdir(idx_files_dir)

    tfrec_path_list = []
    idx_path_list = []
    n_samples = 0
    # Create idx files and create TFRecordPipelines
    for tf_file in tf_files:
        # Path of tf_file and idx file
        tfrec_path = os.path.join(root, split, 'data', tf_file)
        tfrec_path_list.append(tfrec_path)
        idx_path = os.path.join(idx_files_dir, tf_file + '_idx')
        idx_path_list.append(idx_path)
        # Create idx file for tf_file by calling tfrecord2idx script
        if not os.path.isfile(idx_path):
            call(["tfrecord2idx", tfrec_path, idx_path])
        with open(idx_path, 'r') as f:
            n_samples += len(f.readlines())

    pipe = dali.pipeline.Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    with pipe:
        input = fn.readers.tfrecord(
            path=tfrec_path_list,
            index_path=idx_path_list,
            features={"image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
                      "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1)},
            num_shards=num_gpus,
            shard_id=device_id,
            random_shuffle=augment
        )
        images = input["image/encoded"]
        labels = input["image/class/label"] - 1

        # Specify devices to use
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'

        if augment:
            images = dali.fn.decoders.image_random_crop(images, device=decoder_device, output_type=types.RGB,
                                                        random_aspect_ratio=[3. / 4., 4. / 3.], random_area=[0.08, 1.0],
                                                        num_attempts=100)
            # images = dali.fn.decoders.image(images, device=decoder_device, output_type=types.RGB)
            images = dali.fn.resize(images, device=dali_device,
                                    resize_x=224,
                                    resize_y=224,
                                    interp_type=types.INTERP_TRIANGULAR)
            coin = dali.fn.random.coin_flip(probability=0.5)
            images = dali.fn.crop_mirror_normalize(images, device=dali_device,
                                                   dtype=types.FLOAT,
                                                   output_layout=types.NCHW,
                                                   mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                   std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                                   mirror=coin
                                                   )
        else:
            images = dali.fn.decoders.image(images, device=decoder_device, output_type=types.RGB)
            images = dali.fn.resize(images, device=dali_device, resize_shorter=256, interp_type=types.INTERP_TRIANGULAR)

            images = dali.fn.crop_mirror_normalize(images, device=dali_device,
                                                   dtype=types.FLOAT,
                                                   output_layout=types.NCHW,
                                                   mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                   std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                                   crop=(224, 224))

        pipe.enable_api_check(True)
        pipe.set_outputs(images, labels.gpu())

    pipe.build()
    dataloader = DALIClassificationIterator(pipelines=pipe,
                                            reader_name=[key for key in pipe.reader_meta()][0],
                                            auto_reset=True)
    return dataloader
