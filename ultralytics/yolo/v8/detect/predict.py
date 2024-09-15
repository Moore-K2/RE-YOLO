# Ultralytics YOLO 🚀, AGPL-3.0 license
import os

import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops


class DetectionPredictor(BasePredictor):

    def preprocess(self, img):
        """Convert an image to PyTorch tensor and normalize pixel values."""
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    # model = cfg.model or 'yolov8n.pt'
    # source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
    #     else 'https://ultralytics.com/images/bus.jpg'

    # / home / RM_luo / Documents / ultralytics / ultralytics / yolo / v8 / detect
    from pathlib import Path
    detect = Path(os.getcwd())
    device = '0'
    model = detect / 'dota_v1_5cls_20240716/yolov8s_1024/weights/best.pt'
    # print(model)  # ROOT = ultralytics/ultratlytics/    + P0170__1__942___0.jpg
    source = cfg.source if cfg.source is not None else ROOT / 'assets/dota_5cls_val_images'
    save_project = 'detetct_dota_v1_5cls_20240716'
    name = 'detect_' + 'yolov8s_1024'
    save_txt = False
    save_conf = False
    show_labels = True
    show_conf = True
    data = 'dota_v1_5cls.yaml'
    visualize_feature_map = False
    line_thickness = 1
    imgsz = 1024
    args = dict(model=model, source=source, project=save_project, show_conf=show_conf, show_labels=show_labels,
                line_thickness=line_thickness, name=name, device=0,imgsz=imgsz,
                save_txt=save_txt, save_conf=save_conf, data=data, visualize=visualize_feature_map)

    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()
