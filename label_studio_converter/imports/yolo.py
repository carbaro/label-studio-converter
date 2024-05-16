import os
import re
import json  # better to use "imports ujson as json" for the best performance
import pandas as pd
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
logging.basicConfig(level=logging.INFO)
import uuid
from pathlib import Path

from PIL import Image
from typing import Optional, Tuple
from urllib.request import (
    pathname2url,
)  # for converting "+","*", etc. in file paths to appropriate urls

from label_studio_converter.utils import ExpandFullPath
from label_studio_converter.utils import LOCAL_FILES_DOCUMENT_ROOT
from label_studio_converter.imports.label_config import generate_label_config

SKW_PICKLES = Path(os.environ['SKW_PICKLES']) if 'SKW_PICKLES' in os.environ else Path(r'F:\AI\SKW\DroneSurveillance\detection_dvc\detect\scores')

logger = logging.getLogger('yolo')

def validate_alternative_image_path(alt_images_directory):
    try:
        is_relative = Path(alt_images_directory).relative_to(LOCAL_FILES_DOCUMENT_ROOT)
        if is_relative:
            with logging_redirect_tqdm():
                logger.info(f"The alternative directory is valid. It is relative to {LOCAL_FILES_DOCUMENT_ROOT}")
        else:
            with logging_redirect_tqdm():
                logger.error(f"The alternative directory {alt_images_directory} must start with LOCAL_FILES_DOCUMENT_ROOT={LOCAL_FILES_DOCUMENT_ROOT} and must be a child, e.g.: {Path(LOCAL_FILES_DOCUMENT_ROOT) / 'project_images'}")
    except Exception as e:
        with logging_redirect_tqdm():
                logger.error(f"The alternative directory {alt_images_directory} must start with LOCAL_FILES_DOCUMENT_ROOT={LOCAL_FILES_DOCUMENT_ROOT} and must be a child, e.g.: {Path(LOCAL_FILES_DOCUMENT_ROOT) / 'project_images'}")


def convert_yolo_to_ls(
    input_dir,
    out_file,
    alt_imgs_dir=None,
    to_name='image',
    from_name='label',
    out_type="annotations",
    image_root_url='/data/local-files/?d=',
    image_ext='.jpg,.jpeg,.png',
    image_dims: Optional[Tuple[int, int]] = None,
    DJI = True,
    full_imgID = False,
    pickled_fn=None
):
    """Convert YOLO labeling to Label Studio JSON
    :param input_dir: directory with YOLO where images, labels, notes.json are located
    :param out_file: output file with Label Studio JSON tasks
    :param alt_imgs_dir: alternative directory containing images in the same structure as input_dir to handle yolo predictions of type save=False. 
    :param to_name: object name from Label Studio labeling config
    :param from_name: control tag name from Label Studio labeling config
    :param out_type: annotation type - "annotations" or "predictions"
    :param image_root_url: root URL path where images will be hosted, e.g.: http://example.com/images
    :param image_ext: image extension/s - single string or comma separated list to search, eg. .jpeg or .jpg, .png and so on.
    :param image_dims: image dimensions - optional tuple of integers specifying the image width and height of *all* images in the dataset. Defaults to opening the image to determine it's width and height, which is slower. This should only be used in the special case where you dataset has uniform image dimesions.
    :param DJI:
    :param full_imgID:
    :param pickled_fn: 
    """

    is_DJI = DJI
    has_alt_imgs_dir = alt_imgs_dir is not None
    validate_alternative_image_path(alt_imgs_dir)
    if full_imgID:
        logger.info('Using full image IDs')
    else:
        logger.info('Using ints image IDs --  assume unique int IDs')
    tasks = []
    with logging_redirect_tqdm():
        logger.info('Reading YOLO notes and categories from %s', input_dir)


    # build categories=>labels dict
    notes_file = os.path.join(input_dir, 'classes.txt')
    with open(notes_file) as f:
        lines = [line.strip() for line in f.readlines()]
    categories = {i: line for i, line in enumerate(lines)}
    with logging_redirect_tqdm():
        logger.info(f'Found {len(categories)} categories')

    # generate and save labeling config
    label_config_file = out_file.replace('.json', '') + '.label_config.xml'
    generate_label_config(
        categories,
        {from_name: 'RectangleLabels'},
        to_name,
        from_name,
        label_config_file,
    )

    # define directories
    labels_dir = os.path.join(input_dir, 'labels') if not has_alt_imgs_dir else input_dir
    images_dir = os.path.join(input_dir, 'images') if not has_alt_imgs_dir else alt_imgs_dir
    get_int = lambda s: int(re.search(r'\d{4}\.',s)[0][:4]) if re.search(r'\d{4}\.',s) is not None else Path(s).stem# s[19:23])
    get_int = lambda s: int(re.search(r'\d{4}\.',s)[0][:4]) if re.search(r'\d{4}\.',s) is not None else Path(s).stem# s[19:23])

    with logging_redirect_tqdm():
        logger.info('Converting labels from %s', labels_dir)

    # build array out of provided comma separated image_extns (str -> array)
    image_ext = [x.strip() for x in image_ext.split(",")]
    with logging_redirect_tqdm():
        logger.info(f'image extensions->, {image_ext}')

    img_count = sum(len([file for file in files if file.endswith('.JPG')]) for _, _, files in os.walk(images_dir))  # Get the number of files

    property = Path(alt_imgs_dir).stem
    pickles = [f for f in os.listdir(SKW_PICKLES) if f.endswith('.pkl')]
    if pickled_fn is None:
        print(f'Pickle filename not provided. Selecting first instance in list')
        for pickle_fn in pickles: 
            if property in pickle_fn and 'all' in pickle_fn:
                property_pickle_fn = pickle_fn
                break
    else:
        print(f'Pickle filename provided. Searching for match in list')
        if pickled_fn in pickles:
            property_pickle_fn = pickles[pickles.index(pickled_fn)]
        else:
            print(f'can\'t find pickle {pickled_fn} in {SKW_PICKLES}')
    if is_DJI: # TODO: files -> images = [file if file.endswith('.JPG)]
        df = pd.read_pickle(SKW_PICKLES / property_pickle_fn)
        with logging_redirect_tqdm():
            logger.info(f'Loading pickled DF {property_pickle_fn}')
        index_names = ['Paddock','FlightID','droneID']
        if df.index.names != index_names:
            df.set_index(index_names, inplace=True)
        df_idx = df.index
        with logging_redirect_tqdm():
            logger.info(f'loaded {property_pickle_fn} with {df.shape[0]} records. Query it by {df.index.names}. e.g. {df.index[0]}')

    with tqdm(total=img_count) as pbar:
    # loop through images
        with logging_redirect_tqdm():
            logger.info('scanning images at %s', images_dir)
        for root, dirs, files in os.walk(images_dir):
                # property = Path(alt_imgs_dir).stem
                # for pickle_fn in [f for f in os.listdir(SKW_PICKLES) if f.endswith('.pkl')]: # TODO: move out of walk. Grab property from alt_imgs_dir
                #     if property in pickle_fn and 'all' in pickle_fn:
                #         property_pickle_fn = pickle_fn
                #         break
                # df = pd.read_pickle(SKW_PICKLES / property_pickle_fn)
                # with logging_redirect_tqdm():
                #     logger.info(f'Loading pickled DF {property_pickle_fn}')
                # index_names = ['Paddock','FlightID','droneID']
                # if df.index.names != index_names:
                #     df.set_index(index_names, inplace=True)
                # df_idx = df.index
                # with logging_redirect_tqdm():
                #     logger.info(f'loaded {property_pickle_fn} with {df.shape[0]} records. Query it by {df.index.names}. e.g. {df.index[0]}')
                
            for f in files:
                pbar.update(1)
                image_file_found_flag = False
                for ext in image_ext:
                    if f.endswith(ext):
                        image_filename = f
                        image_file_base = f[0 : -len(ext)]
                        image_file_found_flag = True
                        break
                if not image_file_found_flag:
                    continue

                if has_alt_imgs_dir:
                    relative_root_pth = Path(root).relative_to(Path(LOCAL_FILES_DOCUMENT_ROOT))
                    relative_root = '/'.join(relative_root_pth.parts)
                image_root_url += '' if image_root_url.endswith('/') else '/'
                task = {
                    "data": {
                        # eg. '../../foo+you.py' -> '../../foo%2Byou.py'
                        "image":    image_root_url
                    + str(pathname2url(image_filename)) if not has_alt_imgs_dir else 
                                    image_root_url 
                                    + relative_root + '/'
                                    + str(pathname2url(image_filename)),
                        "storage_filename": image_filename
                    }
                }
                if is_DJI:
                    if len(relative_root_pth.parts) <3:
                        property, paddock, flight, *_ = list(relative_root_pth.parts) + [None,None]
                    else:
                        property, paddock, flight = relative_root_pth.parts[-3:]
                        flight = flight[-3:] if all(c.isdigit() for c in flight[-3:]) else flight
                    # try:
                        # img_id = int(Path(image_filename).stem.split('_')[2])
                    img_id = image_file_base if full_imgID else get_int(image_filename)
                    # except:
                        # logger.info(f'img filename: {image_filename}')
                    task['data']['property'] = property
                    task['data']['paddock'] = paddock
                    task['data']['flight'] = flight
                    pic_midx = (paddock,flight,img_id)
                    # with logging_redirect_tqdm():
                    if pic_midx in df_idx:# Detection data
                        # with logging_redirect_tqdm():
                        #     logger.info(f'looking for {pic_midx} in {df_idx}')
                        pic_df = df.loc[pic_midx]
                        # task['data']['max'] = pic_df['c'].max() # if loading all dets
                        task['data']['max'] = pic_df['max'] # if loading summary per pic
                        # task['data']['max'] = pic_df['mean'] # if loading summary per pic
                        # task['data']['max'] = pic_df['count'] # if loading summary per pic
                    elif full_imgID and img_id in df_idx.unique(level='droneID'): #(paddock is None or flight is None) and img_id in df_idx.unique(level='droneID'):
                        # with logging_redirect_tqdm():
                        # with logging_redirect_tqdm():
                        #     logger.info(f'looking for [:,:,{img_id}]')# in {df_idx}')
                        pic_df = df.loc[:,:,img_id]
                        # with logging_redirect_tqdm():
                        # with logging_redirect_tqdm():
                        #     logger.info(f'found {pic_df}')
                        task['data']['max'] = pic_df['max'].tolist()[0] # if loading summary per pic
                    else:
                        # logger.info(f'{img_id} could not be retrieved')# from {df}')
                        pass

                # define coresponding label file and check existence
                label_file = os.path.join(labels_dir, image_file_base + '.txt') if not has_alt_imgs_dir else os.path.join(root.replace(images_dir,labels_dir), 'labels', image_file_base + '.txt')
                # logger.info(f'label file is {label_file}')

                if os.path.exists(label_file):
                    task[out_type] = [
                        {
                            "result": [],
                            "ground_truth": False,
                        }
                    ]

                    # read image sizes
                    if image_dims is None:
                        # default to opening file if we aren't given image dims. slow!
                        image_path = os.path.join(images_dir, relative_root, image_filename) if not has_alt_imgs_dir else os.path.join(root, image_filename)
                        with Image.open(image_path) as im:
                            image_width, image_height = im.size
                    else:
                        image_width, image_height = image_dims

                    with open(label_file) as file:
                        # convert all bounding boxes to Label Studio Results
                        lines = file.readlines()
                        for line in lines:
                            label_id, x, y, width, height = line.split()[:5]
                            conf = line.split()[-1] if out_type == 'predictions' else None
                            x, y, width, height = (
                                float(x),
                                float(y),
                                float(width),
                                float(height),
                            )
                            conf = float(conf) if conf is not None else None
                            item = {
                                "id": uuid.uuid4().hex[0:10],
                                "type": "rectanglelabels",
                                "value": {
                                    "x": (x - width / 2) * 100,
                                    "y": (y - height / 2) * 100,
                                    "width": width * 100,
                                    "height": height * 100,
                                    "rotation": 0,
                                    "rectanglelabels": [categories[int(label_id)]],
                                },
                                "to_name": to_name,
                                "from_name": from_name,
                                "image_rotation": 0,
                                "original_width": image_width,
                                "original_height": image_height,
                            }
                            if out_type == 'predictions':
                                item["score"] = conf
                            task[out_type][0]['result'].append(item)

                tasks.append(task)

    if len(tasks) > 0:
        with logging_redirect_tqdm():
            logger.info('Saving Label Studio JSON to %s', out_file)
        with open(out_file, 'w') as out:
            json.dump(tasks, out)

        logger.info(
            '\n'
            f'  1. Create a new project in Label Studio\n'
            f'  2. Use Labeling Config from "{label_config_file}"\n'
            f'  3. Setup serving for images [e.g. you can use Local Storage (or others):\n'
            f'     https://labelstud.io/guide/storage.html#Local-storage]\n'
            f'  4. Import "{out_file}" to the project\n'
        )
    else:
        with logging_redirect_tqdm():
            logger.error('No labels converted')


def add_parser(subparsers):
    yolo = subparsers.add_parser('yolo')

    yolo.add_argument(
        '-i',
        '--input',
        dest='input',
        required=True,
        help='directory with YOLO where images, labels, notes.json are located',
        action=ExpandFullPath,
    )
    yolo.add_argument(
        '-o',
        '--output',
        dest='output',
        help='output file with Label Studio JSON tasks',
        default='output.json',
        action=ExpandFullPath,
    )
    yolo.add_argument(
        '-a',
        '--alt-imgs-dir',
        dest='alt_imgs_dir',
        required=False,
        help='Alternative Directory containing images',
        action=ExpandFullPath,
    )
    yolo.add_argument(
        '--to-name',
        dest='to_name',
        help='object name from Label Studio labeling config',
        default='image',
    )
    yolo.add_argument(
        '--from-name',
        dest='from_name',
        help='control tag name from Label Studio labeling config',
        default='label',
    )
    yolo.add_argument(
        '--out-type',
        dest='out_type',
        help='annotation type - "annotations" or "predictions"',
        default='annotations',
    )
    yolo.add_argument(
        '--image-root-url',
        dest='image_root_url',
        help='root URL path where images will be hosted, e.g.: http://example.com/images',
        default='/data/local-files/?d=',
    )
    yolo.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image extension to search: .jpeg or .jpg, .png',
        default='.jpg',
    )
    yolo.add_argument(
        '--image-dims',
        dest='image_dims',
        type=int,
        nargs=2,
        help=(
            "optional tuple of integers specifying the image width and height of *all* "
            "images in the dataset. Defaults to opening the image to determine it's width "
            "and height, which is slower. This should only be used in the special "
            "case where you dataset has uniform image dimesions. e.g. `--image-dims 600 800` "
            "if all your images are of dimensions width=600, height=800"
        ),
        default=None,
    )
    yolo.add_argument(
        '--full-imgID',
        dest='full_imgID',
        type=bool,
        default=False
    )
    yolo.add_argument(
        '--pickled-fn',
        dest='pickled_fn',
        type=str,
        default=None
    )
