# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# reference: https://github.com/sukjunhwang/IFC/blob/master/projects/IFC/demo/predictor.py
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode

from panopticapi.utils import IdGenerator, rgb2id


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = VideoPredictor(cfg)
        
        categories = {el['id']: el for el in self.metadata.categories}
        self.color_generator = IdGenerator(categories)
        self.VOID = -1
        self.inst2color = {}

        if self.metadata.name =='panoVSPW_vps_video_val':
            self.div_ = 125
        else:
            raise NotImplementedError

    def run_on_video(self, frames):
        """
        Args:
            frames (List[np.ndarray]): a list of images of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        visualizations = []
        panoptic_segs = self.predictor(frames)[0][0][0]
        
        T = len(panoptic_segs)
        for t in range(T):
            img_tensor = panoptic_segs[t].cpu().numpy()
            pan_format = np.zeros((img_tensor.shape[0], img_tensor.shape[1], 3), dtype=np.uint8)
            l = np.unique(img_tensor)
            for el in l:
                if el == self.VOID:
                    continue
                mask = img_tensor == el

                if el < self.div_:
                    if el in self.metadata.thing_dataset_id_to_contiguous_id.values():
                        continue
                    else:
                        sem = el
                        if el in self.inst2color:
                            color = self.inst2color[el]

                        else:
                            color = self.color_generator.get_color(sem)
                            self.inst2color[el] = color

                else:
                    sem = int(el)//int(self.metadata.label_divisor)

                    if sem not in self.metadata.thing_dataset_id_to_contiguous_id.values():
                        print(l)
                        print('error')
                        print(sem)
                        print(el)
                        exit()
                    if el in self.inst2color:
                        color = self.inst2color[el]
                    else:
                        color = self.color_generator.get_color(sem)
                        self.inst2color[el] = color

                pan_format[mask] = color
            visualizations.append(pan_format)
        return visualizations
            
class VideoPredictor(DefaultPredictor):
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """
    def __call__(self, frames):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            input_frames = []
            is_real_pixels = []
            height = []
            width = []
            for original_image in frames:
                # Apply pre-processing to image.
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                image_height, image_width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                is_real_pixel = torch.ones(image.shape)
                input_frames.append(image)
                is_real_pixels.append(is_real_pixel)
                height.append(image_height)
                width.append(image_width)

            inputs = {"image": input_frames, "is_real_pixels": is_real_pixels, "height": height, "width": width}
            predictions = self.model([inputs])
            return predictions


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = VideoPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5